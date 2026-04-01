
###########
# Bequest #
###########

# Backward #
#----------#
function get_V_bequest_i(wealth_next, h, α, price_index, q, bequest_motive, ϕ)

    wealth_postsell = get_wealth_postsell(wealth_next, h, q, ϕ)
    wealth_postsell = 1 + max(wealth_postsell, 0) # Ensure non-negative wealth for the algorithm to work

    #Flexible RRA: α * bequest_motive * (wealth_postsell^(1-η) * price_level_factor - 1) / (1-η)
    #return bequest_motive*log( α * wealth_postsell / price_index )
    return bequest_motive*log( wealth_postsell / price_index )
end

function get_V_bequest!(V_end, prices, params)
    (;ρ, q) = prices
    (;γ, σ, α, bequest_motive, ϕ) = params

    wealth_grid = get_wealth_grid(V_end)
    h_grid = get_h_grid(V_end)

    price_index = @. get_price_index(ρ, γ, σ)
    return V_end .= get_V_bequest_i.(wealth_grid, h_grid, α, price_index, q, bequest_motive, ϕ)
end

get_V_bequest(prices::Prices, params::Params) = get_V_bequest!(zeros(FLOAT, STATE_IDXs), prices, params)
get_V_bequest(prices::PricesGPU, params) = get_V_bequest!(CUDA.zeros(STATE_IDXs), prices, params)
