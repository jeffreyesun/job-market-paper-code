
######################
# House Price Change #
######################

# Wealth #
#--------#
get_wealth_postprice(wealth, h, q, q_last) = wealth + (q - q_last)*h

# Backward #
#----------#
"""
Compute pre-price change value, V_price.

Simply interpolate post-price change value V_preprice onto "post-price
wealth in terms of pre-price wealth," wealth_postprice_k_preprice to
get "pre-price value in terms of pre-price wealth."
"""
function get_V_price(V_postprice, prealloc)
    (;wealth_postprice_k_preprice) = prealloc
    V_price = prealloc.V_start

    apply_wealth_change_V!(V_price, V_postprice, wealth_postprice_k_preprice)
    #validate_V_price(V_price)
    return V_price
end

# Forward #
#---------#
function get_λ_postprice(λ_preprice, prealloc)
    (;λ_postprice, wealth_postprice_k_preprice) = prealloc
    return apply_wealth_change_λ!(λ_postprice, λ_preprice, wealth_postprice_k_preprice)
end
