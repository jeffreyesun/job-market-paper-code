
############
# Buy Home #
############

# Backward #
#----------#
function get_V_choosebuy(V_postbuy, prealloc)
    (;V_choosebuy) = prealloc

    V_choosebuy .= V_postbuy
    V_choosebuy[:,:,1:1,:] .= maximum(V_postbuy; dims=H_DIM)
    return V_choosebuy
end

# Forward #
#---------#
function get_P_buy(V_postbuy, prealloc, params; taste_shocks=true)
    (;P_buy) = prealloc
    (;ξ) = params

    @. P_buy = exp(ξ*V_postbuy - ξ*V_postbuy[:,:,1:1,:])
    @inbounds P_buy ./= sum(P_buy; dims=H_DIM)
    
    if !taste_shocks
        @error "Not yet implemented."
        P_buy .= 0
        h_choice = argmax(V_postbuy; dims=H_DIM)
        P_buy[h_choice] .= 1
    end
    assert_nonan(P_buy)
    return P_buy
end

function get_λ_postbuy(λ_prebuy, prealloc, params)
    (;λ_postbuy) = prealloc

    P_buy = get_P_buy(prealloc.V_income, prealloc, params)

    # Start by adding all existing homeowners to the postbuy population
    @views λ_postbuy[:,:,2:end,:] .= λ_prebuy[:,:,2:end,:]
    
    # Then add buyers to the corresponding post-buy states. (Note that choosing not to buy is hi=1.)
    @views λ_postbuy[:,:,1,:] .= 0
    return @. λ_postbuy += λ_prebuy[:,:,1:1,:] * P_buy
end
