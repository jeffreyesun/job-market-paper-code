
"""
Methods to solve the within-period model as a single stage. That is, given boundary conditions (Λ_start, V_end).
"""


######################
# Solve Given Prices #
######################

function iterate_V_backward!(pd::Union{PeriodData,PeriodDataGPU}, V_end, params)
    (;prices) = pd
    assert_approx(V_end[end], get_V_bequest(prices, params))
    @assert length(V_end) == N_AGE

    #@threads 
    for agei in 1:N_AGE
        iterate_V_backward!(pd[agei], V_end[agei], params)
    end
    return pd
end

function iterate_λ_forward!(pd::Union{PeriodData,PeriodDataGPU}, λ_start, params)
    assert_approx(λ_start[1], get_λ_born_based_on_V(pd.V_start[1], params))
    @assert length(λ_start) == N_AGE    

    #@threads 
    for agei=1:N_AGE
        iterate_λ_forward!(pd[agei], λ_start[agei], params)
    end

    return pd
end

"""
Given prices, solve a period ``stage-style'', that is,
(Λ_start, V_end) |> (Λ_end, V_start).
"""
function solve_as_stage_without_market_clearing!(pd, q, ρ, Λ_start, V_end_surviving, params::Params)
    (;λ_start_surviving, H_S_prior, q_last) = Λ_start

    set_prices_and_params!(pd, params; q, ρ, q_last, H_S_prior)

    # Iterate V backward
    V_end = [V_end_surviving..., get_V_bequest(pd.prices, params)]
    iterate_V_backward!(pd, V_end, params)

    # Iterate λ forward
    λ_start = [get_λ_born_based_on_V(pd.V_start[1], params), λ_start_surviving...]
    iterate_λ_forward!(pd, λ_start, params)

    (;V_start, λ_end, q) = pd
    Λ_end = (;λ_end, H_S=params.H_S, q)
    return (;pd, V_start, Λ_end)
end


####################
# Solve For Prices #
####################

"""
Get the excess demand, which implicitly defines prices, given a guess of
prices and boundary conditions (Λ_start, V_end). 
"""
function get_excess_demand!(pd, q, ρ, Λ_start, V_end, params::Params)
    solve_as_stage_without_market_clearing!(pd, q, ρ, Λ_start, V_end, params)

    H_rent = get_H_rent(pd, params)
    H_D = get_H_D(pd)
    H_S = params.H_S
    
    H_rent_err = @. (H_rent-H_D)/H_D
    H_D_err = @. (H_D-H_S)/H_S
    
    return (;H_rent_err, H_D_err)
end

"""
Given boundary conditions (Λ_start, V_end), solve for prices and (Λ_end, V_start).
That is, solve the within-period problem, possibly off-steady-state.
"""
function _solve_as_stage!(pd, V_end, Λ_start, params::Params; update_speed=50, rtol=1e-5, verbosity=0, q_init=nothing, ρ_init=nothing)

    q_guess = copy(isnothing(q_init) ? pd.q : q_init)
    ρ_guess = copy(isnothing(ρ_init) ? pd.ρ : ρ_init)

    n = 0
    err = Inf
    while err > rtol
        (;H_rent_err, H_D_err) = get_excess_demand!(pd, q_guess, ρ_guess, Λ_start, V_end, params)

        update_speed_cos = update_speed * cos(n/100)^2 

        @. q_guess += update_speed_cos * H_rent_err * q_guess/ρ_guess*0.5
        @. q_guess += update_speed_cos * H_D_err * 1
        @. ρ_guess += update_speed_cos * H_rent_err * 1

        err = rms(H_rent_err) + rms(H_D_err)
        verbosity > 0 && @printf "%e %e %e\n" rms(H_rent_err) rms(H_D_err) err
        n += 1
    end
    
    solve_as_stage_without_market_clearing!(pd, q_guess, ρ_guess, Λ_start, V_end, params)
    return pd
end

# Wrap according to value of ACTIVELY_DISPATCH_TO_GPU #
#-----------------------------------------------------#
function solve_as_stage!(pd, V_end, Λ_start, params::Params; q_init=nothing, ρ_init=nothing, kwargs...)

    # Run on GPU if ACTIVELY_DISPATCH_TO_GPU == true
    if ACTIVELY_DISPATCH_TO_GPU && pd isa PeriodData
        # Move inputs to GPU
        pd_g = to_gpu(pd)
        params_g = to_gpu(params)
        V_end_g = gpu(V_end)
        Λ_start_g = gpu(Λ_start)
        q_init_g = isnothing(q_init) ? nothing : gpu(q_init)
        ρ_init_g = isnothing(ρ_init) ? nothing : gpu(ρ_init)

        # Solve on GPU
        _solve_as_stage!(pd_g, V_end_g, Λ_start_g, params_g; q_init=q_init_g, ρ_init=ρ_init_g, kwargs...)

        # Move output back to CPU
        pd = to_cpu!(pd, pd_g)
        params = to_cpu!(params, params_g)
    else
        _solve_as_stage!(pd, V_end, Λ_start, params; q_init, ρ_init, kwargs...)
    end

    return pd
end


##############################
# API For Forward Simulation #
##############################

"Get the start-of-next-period state Λ_start_{t+1}, given period-t data `pd` and parameters `params`."
function get_Λ_start_next(pd::PeriodData, params::Params) #TODO Rename construct_Λ_start_next or just factor out entirely
    (;λ_end, q) = pd
    (;H_S) = params

    λ_start_surviving = λ_end[1:end-1]
    return Λ_start_next = (;λ_start_surviving, H_S_prior=H_S, q_last=q)
end

"Solve the within-period model as a single stage, given last-period data `pd_last` and parameters `params_last`."
function solve_as_stage!(pd::PeriodData, pd_last::PeriodData, params_last::Params, V_end, params::Params; kwargs...)

    return solve_as_stage!(pd, V_end, get_Λ_start_next(pd_last, params_last), params; kwargs...)
end
