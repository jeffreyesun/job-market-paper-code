
###############################
# Steady State Model Solution #
###############################

# Backward #
#----------#
function solve_V_backward_steady_state!(pd::Union{PeriodData, PeriodDataGPU}, params::Params)
    (;prices) = pd
    V_end = get_V_bequest!(pd.V_end[end], prices, params)

    # Iterate value function backward
    for agei in reverse(1:N_AGE)
        global agei_last = agei
        V_end, period_solution_slice = iterate_V_backward!(pd[agei], V_end, params)
    end

    return pd
end

# Forward #
#---------#
function simulate_λ_forward_steady_state!(pd::Union{PeriodData, PeriodDataGPU}, params::Params)
    λ_start = get_λ_born_based_on_V(pd.V_start[1], params)

    # Iterate state distribution forward
    for agei in 1:N_AGE
        global agei_last = agei
        λ_start = iterate_λ_forward!(pd[agei], λ_start, params)
    end
    
    return pd
end


#############
# Interface #
#############

"""
Solve the household's problem in steady-state, given prices. Yields the household's full
value function and state distribution. Market-clearing conditions are not enforced.
"""
function solve_household_problem_steady_state!(period_data, params, prices=period_data.prices; ρ=prices.ρ, q=prices.q, H_S=nothing)

    set_prices_and_params!(period_data, params; q, ρ, q_last=q, H_S, H_S_prior=0)
    solve_V_backward_steady_state!(period_data, params)
    return pd = simulate_λ_forward_steady_state!(period_data, params)
end
solve_household_problem_steady_state(prices, params) = solve_household_problem_steady_state!(PeriodData(prices, params), params)
