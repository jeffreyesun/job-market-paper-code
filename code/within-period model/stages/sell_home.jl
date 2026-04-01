
#############
# Sell Home #
#############

# Wealth #
#--------#
"Get post-move wealth in terms of pre-move wealth."
@inline get_wealth_postsell(wealth_presell, h, q, ϕ) = wealth_presell - ϕ*q*h

# Backward #
#----------#
function get_V_choosesell(V_postsell, prealloc)
    (;wealth_postsell_k_presell, V_sell, V_choosesell) = prealloc

    V_sell_postsell = V_postsell[:,:,1:1,:]
    V_nosell = V_postsell
    
    # Compute the value of selling old h, using the value prior to buying new h as the continuation value
    # If households are bankrupt after selling, set their wealth to the minimum
    V_sell = @views apply_wealth_change_V!(V_sell, V_sell_postsell, wealth_postsell_k_presell, Val(:clip))
    
    # Compute the value prior to selling, maximizing over choices
    @. V_choosesell = max(V_nosell, V_sell)
    return V_choosesell
end

# Forward #
#---------#
function get_λ_postsell(λ_presell, prealloc, params)
    (;V_sell, P_sell, λ_postsell) = prealloc
    (;ξ) = params

    V_nosell = prealloc.V_move

    P_sell[:,:,1,:] .= 0
    @inbounds @simd for idx in CartesianIndices((N_K, N_Z, 2:N_H, N_LOC))
        eV_sell = exp(ξ*(V_sell[idx] - 40))
        P_sell[idx] = eV_sell / (eV_sell + exp(ξ*(V_nosell[idx] - 40)))
    end    

    # Add P_sell share of homeowners to the postsell renter population
    # and (1-P_sell) to the postsell homeowner population
    for idx=CartesianIndices((N_K, N_Z, N_H, N_LOC))
        λ_postsell[idx] = λ_presell[idx] * (1 - P_sell[idx])
        λ_postsell[min(idx,RENT_BOUND)] += λ_presell[idx] * P_sell[idx]
    end
    
    # This replace is slightly dangerous but probably okay because,
    # if the problem is severe, the following assert will fail.
    replace!(λ_postsell, NaN=>0.0)
    assert_sum_approx(λ_postsell, λ_presell)
    return λ_postsell
end


######################
# GPU Implementation #
######################

# Forward #
#---------#
function get_λ_postsell(λ_presell::CuArray, prealloc, params)
    (;V_sell, P_sell, λ_postsell) = prealloc
    (;ξ) = params

    V_nosell = prealloc.V_move

    @. P_sell = exp(ξ*(V_sell - 40))
    @. P_sell = P_sell / (P_sell + exp(ξ*(V_nosell - 40)))
    P_sell[:,:,1,:] .= 0

    # Add P_sell share of homeowners to the postsell renter population
    # and (1-P_sell) to the postsell homeowner population
    @. λ_postsell = λ_presell * (1 - P_sell)
    λ_postsell[:,:,1:1,:] .+= sum(λ_presell .* P_sell, dims=3)
    
    @. λ_postsell = ifelse(isnan(λ_postsell), zero(eltype(λ_postsell)), λ_postsell)
    
    assert_sum_approx(λ_postsell, λ_presell)
    return λ_postsell
end
