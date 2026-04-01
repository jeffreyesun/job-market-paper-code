
########
# Move #
########

# Backward #
#----------#
"""
    Compute V_move in terms of k_postmove.

Because k_postmove is equal regardless of the destination, we can compute
V_move in terms of k_postmove by integrating over all possible destinations.
In fact, this can be a simple matrix multiplication. For each k_postmove, z,
we simply multiply the vector of exp(ψ * V_postmove(ℓ')) by the matrix of
exp(-ψ * F_u(ℓ, ℓ')).
"""
function get_V_move(V_postmove, prealloc, params)
    (;V_move, eψV_postmove_tilde, eψV_move_tilde, eψFu_inv, ψV_means) = prealloc
    (;ψ) = params

    # Subtract A from V_postmove because location definitions are endogenous. High-A locations are smaller,
    # and we need to adjust for this. I'm not sure if there should be a coefficient on A here.
    V_move_postmove = V_postmove[:,:,1:1,:] .+ log.(params.α)
    ψV_means .= ψ.*mean(V_move_postmove; dims=(3,4))
    # Compute exp(ψ * V_postmove)
    @. eψV_postmove_tilde = exp(ψ*V_move_postmove - ψV_means)# - 10)

    assert_all_finite(eψV_postmove_tilde)

    # Reshape everything into one big matrix and multiply
    eψV_postmove_mat = reshape(eψV_postmove_tilde, (N_K*N_Z, N_LOC))
    eψV_move_mat = reshape(eψV_move_tilde, (N_K*N_Z, N_LOC))
    mul!(eψV_move_mat, eψV_postmove_mat, eψFu_inv)

    assert_all_finite(eψV_move_tilde)

    # Recover V_move
    # This represents the location parameter of the Gumbel distribution, not the mean
    V_move .= V_postmove
    @. V_move[:,:,1:1,:] = (log(eψV_move_tilde) + ψV_means)/ψ #+ EULER_GAMMA
    
    assert_nonan(V_move)
    return V_move
end

# Forward #
#---------#
function get_λ_postmove(λ_premove, prealloc)
    (;λ_postmove, eψV_move_tilde, eψFu_inv, eψV_postmove_tilde, λ_move_postmove, origin_weights) = prealloc
    
    @. origin_weights = λ_premove[:,:,1:1,:] / eψV_move_tilde
    
    λ_postmove_mat = reshape(λ_move_postmove, (N_K*N_Z, N_LOC))
    origin_mat = reshape(origin_weights, (N_K*N_Z, N_LOC))
    mul!(λ_postmove_mat, origin_mat, eψFu_inv)
    
    λ_move_postmove .*= eψV_postmove_tilde
    N_LOC == 1 && replace!(λ_move_postmove, NaN=>0)

    λ_postmove .= λ_premove
    λ_postmove[:,:,1:1,:] .= λ_move_postmove[:,:,1:1,:]

    assert_sum_approx(λ_postmove, λ_premove)
    return λ_postmove
end
