
##################
# Receive Income #
##################

# Backward #
#----------#
"""
Compute pre-income value, V_income.

Simply interpolate post-income value V_consume onto "post-income
wealth in terms of pre-income wealth," wealth_postinc_k_preinc.
"""
function get_V_income(V_consume, prealloc, params)
    (;agei, V_income, wealth_postinc_k_preinc, prices) = prealloc
    (;ρ, q) = prices
    (;A, δ, χ, r_m) = params

    # Pre-sale, pre-move `wealth_presell`, in terms of post-sale, post-move `wealth_postmove`
    k_grid = get_wealth_grid(q)
    z_grid = get_z_grid(agei, q)
    h_grid = get_h_grid(q)

    @. wealth_postinc_k_preinc = k_grid + get_income.(k_grid, z_grid, h_grid, A, δ, ρ, q, χ, r_m)

    # We have V_consume in terms of wealth_postinc (= wealth_consume)
    # Put that in terms of wealth_preinc by resampling at wealth_postinc for each wealth_preinc
    apply_wealth_change_V!(V_income, V_consume, wealth_postinc_k_preinc, Val(-Inf))

    @views assert_all_finite(V_income[:,:,1,:])
    return V_income
end

# Forward #
#---------#
function get_λ_postincome(λ_preincome, prealloc)
    (;wealth_postinc_k_preinc, λ_postincome) = prealloc

    apply_wealth_change_λ!(λ_postincome, λ_preincome, wealth_postinc_k_preinc)
    # NOTE This replacement is somewhat dangerous because it could cause things to silently fail
    @. λ_postincome = ifelse(isnan(λ_postincome), zero(eltype(λ_postincome)), λ_postincome)

    return λ_postincome
end
