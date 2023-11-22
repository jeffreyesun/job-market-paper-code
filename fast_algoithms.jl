
##############################################################################################################
# Compute discrete choice (migration) quickly by reducing the entire thing to a single matrix multiplication #
##############################################################################################################

"""
    Iterate value function backwards through migration stage.

Because k_postmove is equal regardless of the destination, we can compute
V_move in terms of k_postmove by integrating over all possible destinations.
In fact, this can be a simple matrix multiplication. For each k_postmove, z,
we simply multiply the vector of exp(ψ * V_postmove(ℓ')) by the matrix of
exp(-ψ * F_u(ℓ, ℓ')).
"""
function get_V_move_k_postmove(V_postmove, prealloc, params)
    (; V_move_k_postmove, eψV_postmove_tilde, eψV_move_k_postmove_tilde, eψFu_inv, ψV_means) = prealloc
    (; ψ) = params

    ψV_means .= ψ.*mean(V_postmove; dims=(3,4,5))
    # Compute exp(ψ * V_postmove)
    @. eψV_postmove_tilde = fastexp(ψ*V_postmove - ψV_means)

    # Reshape everything into one big matrix
    eψV_postmove_mat = reshape(eψV_postmove_tilde, N_K*N_Z, N_LOC)
    eψV_move_k_postmove_mat = reshape(eψV_move_k_postmove_tilde, N_K*N_Z, N_LOC)
    
    mul!(eψV_move_k_postmove_mat, eψV_postmove_mat, eψFu_inv)

    # Recover V_move_k_postmove
    # This represents the location parameter of the Gumbel distribution, not the mean
    @. V_move_k_postmove = fastlog(eψV_move_k_postmove_tilde) + ψV_means# + EULER_GAMMA
    V_move_k_postmove ./= ψ

    return V_move_k_postmove
end

"""
    Simulate state distribution forward through migration stage.
"""
function get_λ_postmove(λ_move_k_postmove, sim_prealloc)
    (;eψV_move_k_postmove_tilde, eψFu_inv, eψV_postmove_tilde, λ_postmove, origin_weights) = sim_prealloc
    
    @. origin_weights = λ_move_k_postmove / eψV_move_k_postmove_tilde

    λ_postmove_mat = reshape(λ_postmove, N_K*N_Z, N_LOC)
    origin_mat = reshape(origin_weights, N_K*N_Z, N_LOC)
    mul!(λ_postmove_mat, origin_mat, eψFu_inv)

    λ_postmove .*= eψV_postmove_tilde
    N_LOC == 1 && replace!(λ_postmove, NaN=>0)
    return λ_postmove
end


############################################################################################################
# Compute optimal consumption-savings decision quickly by solving for entire slice of distribution at once #
############################################################################################################

"""
    For each pre-utility state, maximize utility + post-utility value,
    over the post-utility states.
"""
function k1_argmax!(V_prec, k1, V, u)
    n = length(k1)
    @assert ispow2(n-1)
    k1[1] = 1
    V_prec[1] = u[1][1] + V[1]

    k1[end] = argmax_sum(u[end], V, 1, n)
    V_prec[end] = u[end][k1[end]] + V[k1[end]]
    
    segment_length = div(n-1, 2)
    while segment_length >= 1
        i = 1
        while i < n - 1
            k1_lb = k1[i]
            i += segment_length
            k1_ub = k1[i+segment_length]
            
            k1[i] = argmax_sum(u[i], V, k1_lb, min(k1_ub,i))
            V_prec[i] = u[i][k1[i]] + V[k1[i]]
            
            i += segment_length
        end
        segment_length = div(segment_length, 2)
        #TODO figure out how to do for non-power-of-two vector lengths
    end
    return k1
end

"""
    Solve consumption problem.

Get pre-consumption value V_consume by maximizing
utility + continuation value
over all possible choices of continuation value.
Save the choices as wealthi_postc_k_prec.

V_consume(x) = max_{g,h} u(g,h,ℓ(x)) + V_postconsume(x'(x,g,h))
"""
function get_V_consume(V_postconsume, prealloc)
    (;V_consume, wealthi_postc_k_prec, u_indirect_rent, u_indirect_own) = prealloc

    Threads.@threads for loci=1:N_LOC
        # Compute optimal utility for renters
        u_rent = u_indirect_rent[loci]
        for idx in CartesianIndices((N_Z, 1, N_Hle))
            @views k1_argmax!(
                V_consume[:, idx, loci],
                wealthi_postc_k_prec[:, idx, loci],
                V_postconsume[:, idx, loci],
                u_rent
            )
        end

        # Compute optimal utility for homeowners
        for idx=CartesianIndices((N_Z, 2:N_Hli, N_Hle))
            hlii = idx[2]
            @views k1_argmax!(
                V_consume[:, idx, loci],
                wealthi_postc_k_prec[:, idx, loci],
                V_postconsume[:, idx, loci],
                u_indirect_own[hlii, loci]
            )
        end
    end

    return V_consume, wealthi_postc_k_prec
end

#########################################################################################
# Do value function reinterpolation by solving for entire slice of distribution at once #
#########################################################################################

"Reinterpolate a value function from after a wealth change to before."
function apply_wealth_change_V!(V_pre, V_post, wealth_post, left_extrap=Val(:linear))
    V_pre_tailsize = CartesianIndex(tail(size(V_pre)))
    V_post_tailsize = CartesianIndex(tail(size(V_post)))
    wealth_post_tailsize = CartesianIndex(tail(size(wealth_post)))

    Threads.@threads for idx=CartesianIndices((N_Z, N_Hli, N_Hle, N_LOC))
        @views reinterpolate!(
            V_pre[:,min(idx, V_pre_tailsize)],
            V_post[:,min(idx, V_post_tailsize)],
            WEALTH_GRID_FLAT,
            wealth_post[:,min(idx, wealth_post_tailsize)],
            left_extrap
        )
    end
    return V_pre
end

function reinterpolate!(y2::AbstractVector, y1::AbstractVector, x1::AbstractVector, x2::AbstractVector, left_extrap::Val{:linear}=Val(:linear))
    #@assert issorted(x1) && issorted(x2)
    j = 1
    x1_j1 = x1[2]
    len_x1m1 = length(x1) - 1
    
    #@inbounds
    for i=1:length(x2)
        x2_i = x2[i]
        while x2_i > x1_j1
            if j == len_x1m1
                break
            end
            j += 1
            x1_j1 = x1[j+1]
        end

        x1_j = x1[j]
        y1_j = y1[j]
        y1_j1 = y1[j+1]

        if y1_j == -Inf || y1_j1 == -Inf
            y2[i] = -Inf
        else
            slope = (y1[j + 1] - y1_j) / (x1_j1 - x1_j)
            y2[i] = slope * (x2_i - x1_j) + y1_j
        end
    end
end

#############################################################################################
# Do state distribution reinterpolation by solving for entire slice of distribution at once #
#############################################################################################

"""
    Convert a point mass distribution y defined on a grid x to a
    pointmass distribution y_new defined on a grid x_new.
    Each point mass (x[i], y[i]) is divided between the grid points
    x_new[j-1] < x[i] <= x_new[j].
"""
function convert_distribution!(y_new, y, x, x_new)
    @assert iszero(y[end])
    @assert all(>=(0), y)
    @assert issorted(x) && issorted(x_new)
    y_new .= 0
    
    len_x = length(x)
    len_x_new_minus_1 = length(x_new) - 1

    @inline validate(y_new) = y_new
    # validate(y_new) = validate_converted_distribution(y_new, y)
    
    i = 1
    # Make sure that x_new[1] <= x[i]
    while x[i] < x_new[1]
        y_new[1] += y[i]
        i += 1
        if i == len_x
            #@assert sum(y_new) ≈ sum(y)
            return validate(y_new)
        end
    end
    #@assert x_new[1] <= x[i]
    
    j = 1
    x_new_j_next = x_new[2]
    @inbounds for i=i:len_x
        x_i = x[i]
        # Make sure that x_new[j] <= x[i] < x_new[j+1] <= x_new[end-1]
        while x_i >= x_new_j_next
            j += 1
            if j == len_x_new_minus_1
                y_new[j] += @views sum(y[i:end])
                return validate(y_new)
            end
            x_new_j_next = x_new[j+1]
        end
        #@assert x_new[j] <= x_i < x_new[j+1] <= x_new[end-1]

        left_share = (x_new_j_next - x_i) / (x_new_j_next - x_new[j])
        y_new[j] += y[i] * left_share
        y_new[j+1] += y[i] * (1 - left_share)
    end

    return validate(y_new)
end


function apply_wealth_change_λ!(λ_post, λ_pre, wealth_post)
    wealth_post_tailsize = CartesianIndex(tail(size(wealth_post)))

    Threads.@threads for idx in CartesianIndices((N_Z, N_Hli, N_Hle, N_LOC))
        idx_k = min(idx, wealth_post_tailsize)
        @views convert_distribution!(
            λ_post[:,idx],
            λ_pre[:,idx],
            wealth_post[:,idx_k],
            WEALTH_GRID_FLAT
        )
    end
    return λ_post
end
