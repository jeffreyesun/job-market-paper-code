
#######################
# Consumption-Savings #
#######################

# Backward #
#----------#
"""Solve consumption problem.

Get pre-consumption value V_consume by maximizing
utility + continuation value
over all possible choices of continuation value.
Save the choices as wealthi_postc_k_prec.

V_consume(x) = max_{g,h} u(g,h,ℓ(x)) + V_postconsume(x'(x,g,h))
"""
function get_V_consume(V_postconsume, prealloc)
    (;V_consume, wealthi_postc_k_prec, u_indirect_loc) = prealloc

    # Compute optimal utility for homeowners
    k1_argmax_arr!(V_consume, wealthi_postc_k_prec, V_postconsume, get_log_expenditure_mat(V_postconsume))

    V_consume .+= pad_dims(u_indirect_loc, 3, 0)

    return V_consume
end

# Forward #
#---------#
function get_λ_postc(λ_prec, prealloc)
    (;wealthi_postc_k_prec, λ_postc) = prealloc

    λ_postc .= 0

    @inbounds @simd for idx=CartesianIndices((N_Z, N_H, N_LOC))
        for ki=1:N_K
            k1i = wealthi_postc_k_prec[ki,idx]
            λ_postc[k1i,idx] += λ_prec[ki,idx]
        end
        # Equivalent to @views λ_preshock[[wealthi_postc_k_prec[:,idx],idx]] .+= λ_prec[:,idx]
    end
    
    return λ_postc
end


######################
# GPU Implementation #
######################

# Forward #
#---------#
function get_λ_postc_kernel!(λ_postc, λ_prec, wealthi_postc_k_prec, n_states, n_z, n_h, n_k)
    statei = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if statei <= n_states
        state0 = statei - 1

        zi   = rem(state0, n_z) + 1
        state0 = state0 ÷ n_z

        hi   = rem(state0, n_h) + 1
        loci = state0 ÷ n_h + 1

        @inbounds for ki=1:n_k
            k1i = wealthi_postc_k_prec[ki, zi, hi, loci]
            λ_postc[k1i, zi, hi, loci] += λ_prec[ki, zi, hi, loci]
        end
    end

    return nothing
end

function get_λ_postc(λ_prec::CuArray, prealloc)
    (;wealthi_postc_k_prec, λ_postc) = prealloc

    λ_postc .= 0

    n_k = size(λ_prec, 1)
    n_z = size(λ_prec, 2)
    n_h = size(λ_prec, 3)
    n_loc = size(λ_prec, 4)
    n_states = n_z * n_h * n_loc

    threads = 256
    blocks = cld(n_states, threads)
    CUDA.@cuda threads=threads blocks=blocks get_λ_postc_kernel!(λ_postc, λ_prec, wealthi_postc_k_prec, n_states, n_z, n_h, n_k)

    return λ_postc
end
