
#############################
# Household Model Functions #
#############################

# Utility #
#---------#
#Flexible RRA: get_u(g, h, α, γ, σ, η) = (((1-γ)*(α*g)^(1-σ) + γ*(α*h)^(1-σ))^((1-η)/(1-σ)) - 1)/(1-η)
#get_u(g, h, α, γ, σ) = log( ((1-γ)*(α*g)^((σ-1)/σ) + γ*(α*h)^((σ-1)/σ))^(σ/(σ-1)) )
get_u(g, h, α, γ, σ) = log( ((1-γ)*(g)^((σ-1)/σ) + γ*(h)^((σ-1)/σ))^(σ/(σ-1)) )

get_price_index(ρ, γ, σ) = ( (1-γ)^σ + γ^σ * ρ^(1-σ) )^(1/(1-σ))
get_g_demand(c, ρ, γ, σ) = (1-γ)^σ * c  / get_price_index(ρ, γ, σ)^(1-σ)
get_h_demand(c, ρ, γ, σ) = (γ/ρ)^σ * c  / get_price_index(ρ, γ, σ)^(1-σ)
#Flexible RRA: get_indirect_u(c, ρ, α, γ, σ, η) = ( ((α*c) * get_price_level(ρ, γ, σ)^(σ/(1-σ)) )^(1-η) - 1)/(1-η)
#get_indirect_u(c, ρ, α, γ, σ) = log( α * c / get_price_index(ρ, γ, σ) )
get_indirect_u(c, ρ, α, γ, σ) = log( c / get_price_index(ρ, γ, σ) )

function get_indirect_u(ρ, params)
    (;γ, σ) = params
    (;α) = params.spatial

    #c_factor = @. α*max(WEALTH_GRID - WEALTH_NEXT_GRID, 0)
    c_factor = @. max(WEALTH_GRID - WEALTH_NEXT_GRID, 0)
    return get_indirect_u.(1, ρ, 1, γ, σ) .+ log.(c_factor)
end

# Income #
#--------#

"""
Within the model, it never makes sense for the household to have
positive bonds and positive mortgage, because the mortgage rate
is higher than the bond rate.
"""
function get_interest(wealth, h, q, r_m)

    net_bondholdings = wealth - q*h

    interest_rate = net_bondholdings >= 0 ? r : r_m
    #interest_rate = r_m

    return net_bondholdings * interest_rate
end

function get_noninterest_income(z, h, A, δ, ρ, χ)
    
    income = A*z # earnings
    income += ρ*h # rental income
    #expenses = h*(χ + δ) # Maintenance
    expenses = h*ρ*χ + h*δ # Maintenance

    return income - expenses
end


function get_income(wealth, z, h, A, δ, ρ, q, χ, r_m)
    interest = get_interest(wealth, h, q, r_m)
    noninterest_income = get_noninterest_income(z, h, A, δ, ρ, χ)
    
    return interest + noninterest_income
end

function get_income(wealth, z, h, loc, local_prices, params::Params)
    (;ρ, q) = local_prices
    (;χ, r_m) = params
    (;A, δ) = loc

    return get_income(wealth, z, h, A, δ, ρ, q, χ, r_m)
end


##############################
# Household Problem Solution #
##############################

# Backward #
#----------#
function iterate_V_backward!(prealloc, V_end, params)
    @assert size(V_end) == STATE_IDXs
    V_end = prealloc.V_end .= V_end

    # V_shock
    V_preshock = get_V_preshock(V_end, prealloc)
    assert_renter_V_finite(V_preshock)
    
    # V_consume and consumption/saving
    V_consume = get_V_consume(V_preshock, prealloc)
    assert_renter_V_finite(V_consume)
    
    # V_income
    V_income = get_V_income(V_consume, prealloc, params)
    assert_renter_V_finite(V_income)
    
    # Buy home
    V_choosebuy = get_V_choosebuy(V_income, prealloc)
    assert_renter_V_finite(V_choosebuy)
    # Move
    V_move = get_V_move(V_choosebuy, prealloc, params)
    assert_renter_V_finite(V_move)
    # Sell home
    V_choosesell = get_V_choosesell(V_move, prealloc)
    assert_renter_V_finite(V_choosesell)

    # Price change
    V_start = get_V_price(V_choosesell, prealloc)
    assert_renter_V_finite(V_start)
    
    return V_start, prealloc.period_solution_slice
end

# Forward #
#---------#
function iterate_λ_forward!(prealloc, λ_start, params)
    prealloc.λ_start .= λ_start

    # Price change
    λ_postprice = get_λ_postprice(λ_start, prealloc)

    # Sell home
    λ_postsell = get_λ_postsell(λ_postprice, prealloc, params)

    # Move
    λ_postmove = get_λ_postmove(λ_postsell, prealloc)

    # Buy home
    λ_postbuy = get_λ_postbuy(λ_postmove, prealloc, params)

    # Get population by post-income, pre-consumption wealth
    λ_postincome = get_λ_postincome(λ_postbuy, prealloc)

    # Get population by post-consumption, pre-shock wealth
    λ_postc = get_λ_postc(λ_postincome, prealloc)

    # Get post-shock, end-of-period population
    return λ_postshock = get_λ_postshock(λ_postc, prealloc)
end


#######################
# Get Solved Age Data #
#######################

function apply_period_stage!(ad::AgeData, V_end, λ_start, params::Params, prices::Prices=ad.prices; q=prices.q, ρ=prices.ρ, q_last=prices.q_last, H_S=nothing, H_S_prior=0)
    @warn "`apply_period_stage!` is for testing purposes only and does not fully apply data such as q_last and H_S."
    set_prices_and_params!(ad, params; q, ρ, q_last, H_S, H_S_prior)
    V_start = iterate_V_backward!(ad, V_end, params)
    λ_end = iterate_λ_forward!(ad, λ_start, params)
    #return (;V_start, λ_end, ad)
    return ad
end
function apply_period_stage!(ad::AgeData, params::Params; prices::Prices=ad.prices)
    (;V_end, λ_start) = ad
    return apply_period_stage!(ad, V_end, λ_start, params, prices)
end

function solve_household_problem_age(agei::Int, prices::Prices, params::Params, V_end, λ_start)
    return ad = apply_period_stage!(AgeData(;agei), V_end, λ_start, params; prices)
end

"Extract and solve a given age group, given an already-solved PeriodData."
function get_age_data_solved!(pd::PeriodData, agei::Int, V_end, λ_start, params::Params)
    ad = AgeData(pd, agei)
    iterate_V_backward!(ad, V_end, params)
    iterate_λ_forward!(ad, λ_start, params)
    return ad
end
get_age_data_solved!(pd::PeriodData, agei::Int, params::Params) = get_age_data_solved!(pd, agei, pd.V_end[agei], pd.λ_start[agei], params)
