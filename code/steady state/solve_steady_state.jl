
function get_H_D_and_rent_from_ρ_and_q_steady_state!(pd, params, ρ, q; H_S=nothing, kwargs...)
    pd = solve_household_problem_steady_state!(pd, params; ρ, q, H_S, kwargs...)
    H_rent = get_H_rent(pd, params)
    H_D = get_H_D(pd)
    return (;H_rent, H_D)
end
get_H_D_and_rent_from_ρ_and_q_steady_state!(pd, params; kwargs...) = get_H_D_and_rent_from_ρ_and_q_steady_state!(pd, params, pd.ρ, pd.q; kwargs...)


function _solve_steady_state_prices!(pd, params; verbosity=0, H_S_endogenous=true,
    update_speed=50, rtol=1e-4, ρ_init=pd.prices.ρ, q_init=pd.prices.q, max_iter=10_000
    )

    if H_S_endogenous
        set_prices_and_params!(pd, params; H_S_prior=0)
    else
        set_prices_and_params!(pd, params; H_S=params.H_S)
    end

    (;ρ, q) = pd.prices


    pd_g = to_gpu(pd)
    params_g = to_gpu(params)

    (;H_S) = params_g
    (;ρ, q) = pd_g

    exogenous_H_S = H_S_endogenous ? nothing : H_S

    q .= gpu(deepcopy(q_init))
    ρ .= gpu(deepcopy(ρ_init))

    price_err = Inf
    iterations = 0
    while price_err > rtol
        (;H_rent, H_D) = get_H_D_and_rent_from_ρ_and_q_steady_state!(pd_g, params_g, ρ, q; H_S=exogenous_H_S)

        H_rent_err = (H_rent .- H_D)./H_D
        H_D_err = (H_D .- H_S)./H_S        
        price_err = rms(H_rent_err) + rms(H_D_err)
        
        verbosity > 0 && @printf "H_rent_err %e, H_D_err %e, price_err %e\n" rms(H_rent_err) rms(H_D_err) price_err
        
        iterations += 1
        if iterations == max_iter
            @warn "Solver stuck at tolerance $rtol after $iterations iterations. Returning current prices."
            break
        end
        
        price_err <= rtol && break
        
        update_speed_cos = update_speed# * cos(iterations/100)^2
        @. q += update_speed_cos*H_rent_err*q/ρ*0.5
        @. q += update_speed_cos*H_D_err*1
        @. ρ += update_speed_cos*H_rent_err*1
    end

    verbosity > 0 && @show iterations
    
    to_cpu!(pd, pd_g)
    return (;pd, ρ=pd.ρ, q=pd.q)
end

# Wrap according to value of ACTIVELY_DISPATCH_TO_GPU #
#-----------------------------------------------------#
function solve_steady_state_prices!(pd, params; ρ_init=pd.prices.ρ, q_init=pd.prices.q, kwargs...)

    # Run on GPU if ACTIVELY_DISPATCH_TO_GPU == true
    if ACTIVELY_DISPATCH_TO_GPU && pd isa PeriodData
        # Move inputs to GPU
        pd_g = to_gpu(pd)
        params_g = to_gpu(params)
        ρ_init_g = isnothing(ρ_init) ? nothing : gpu(ρ_init)
        q_init_g = isnothing(q_init) ? nothing : gpu(q_init)

        # Solve on GPU
        _solve_steady_state_prices!(pd_g, params_g; ρ_init=ρ_init_g, q_init=q_init_g, kwargs...)

        # Move output back to CPU
        pd = to_cpu!(pd, pd_g)
        params = to_cpu!(params, params_g)
    else
        _solve_steady_state_prices!(pd, params; ρ_init, q_init, kwargs...)
    end
    return (;pd, ρ=deepcopy(pd.ρ), q=deepcopy(pd.q))
end


#######
# API #
#######

function solve_steady_state_prices(params=get_test_params(); prices_init=get_test_prices(), verbose=false, kwargs...)
    pd = PeriodData(prices_init, params)
    return solve_steady_state_prices!(pd, params; verbose, kwargs...).pd
end

function solve_steady_state!(pd, params; endogenous_entry=false, H_S_endogenous=true, kwargs...)

    # Check that we are asking for specifically a solution for populations and housing stocks, given Π and elasticity
    endogenous_entry && error("Endogenous Entry Not Implemented")
    H_S_endogenous || error("The function `solve_steady_state` assumes endogenous housing supply.")
    (any(isnan, params.spatial.Π) | any(isnan, params.spatial.elasticity)) && error("Housing Supply parameters Π or elasticity not set.")

    return solve_steady_state_prices!(pd, params; H_S_endogenous, kwargs...)
end

function solve_steady_state(params, prices_init=read_prices(); kwargs...)
    pd = PeriodData(prices_init, params)
    return solve_steady_state!(pd, params; kwargs...)
end