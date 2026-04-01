
############
# Simulate #
############

"""
Get the input that should be given to the neural network vnet, consisting of
end-of-last-period household state and this-period climate-sensitive parameters.
"""
function get_vnet_input(Λ_start, params, pop_arrays=POP_ARRAYS)
    (;H_S_prior, q_last, λ_start_surviving) = Λ_start
    (;A, α, δ, elasticity, Π, decade, scenario_ind) = params
    
    loc_mat = stack(vec.([A, α, δ, H_S_prior, q_last, elasticity, Π, fill(decade, N_LOC), fill(scenario_ind, N_LOC)]))'
    loc_mat = vcat(loc_mat, ID_NLOC)

    return (;loc_mat, λ_start_surviving, pop_arrays)
end

"""
Use the neural network vnet to predict V_end given the initial state (Λ_start, params)
"""
function predict_V_end_by_vnet(Λ_start, params, vnet)
    vnet_input_cpu = get_vnet_input(Λ_start, params)
    vnet_input = CUDA.functional() ? gpu(vnet_input_cpu) : vnet_input_cpu
    
    return V_end_pred = eachslice(cpu(vnet(vnet_input)); dims=AGE_DIM)
end

"""
Using vnet as a drop-in replacement for the end-of-period value function, simulate pd forward from the beginning of the period
to the end of the period.
"""
function simulate_forward_given_vnet!(pd::PeriodData, pd_last::PeriodData, params_last::Params, vnet::VNet, params::Params; scenario_ind, V_end=nothing, verbosity=0, kwargs...)
    
    # Set scenario_ind for this period, either by drawing or using the provided value.
    params.scenario_ind = scenario_ind==:draw ? draw_scenario_ind(params_last.scenario_ind) : scenario_ind
    
    # Get boundary conditions which suffice to solve the period
    Λ_start = get_Λ_start_next(pd_last, params_last)
    V_end = @something V_end predict_V_end_by_vnet(Λ_start, params, vnet)

    # Solve the period given those boundary conditions
    pd = solve_as_stage!(pd, V_end, Λ_start, params; verbosity=verbosity-1, kwargs...)

    # Report progress
    verbosity >= 1 && println("Simulated period $(params.decade), scenario_ind=$(params.scenario_ind)")
    return pd
end

"""
Given a neural network vnet approximating the end-of-period value function,
simulate a transition path forward, possibly stochastically, taking one draw over possible transition paths.
"""
function simulate_path_given_vnet!(td::Vector{<:PeriodData}, pp::Vector{<:Params}, vnet::VNet; SST_process=:deterministic, kwargs...)

    T = length(pp)
    @assert SST_process in (:deterministic, :stochastic)
    
    for t=2:T-1
        @assert pp[t].decade == t
        scenario_ind = SST_process==:deterministic ? pp[t].scenario_ind : :draw

        simulate_forward_given_vnet!(td[t], td[t-1], pp[t-1], vnet, pp[t]; scenario_ind, kwargs...)
    end

    return td
end
