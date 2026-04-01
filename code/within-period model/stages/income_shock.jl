
################
# Income Shock #
################

# Backward #
#----------#
"Compute V_preshock. Get values in terms of z by multiplying along z' by z_T"
function get_V_preshock(V_end, prealloc)
    (;V_preshock, V_end_perm, V_preshock_perm) = prealloc

    # Put zi in front, so the entire thing is one big matrix multiplication
    permutedims!(V_end_perm, V_end, (2,1,3,4))

    V_end_mat = reshape(V_end_perm, N_Z, N_K*N_H*N_LOC)
    V_preshock_mat = reshape(V_preshock_perm, N_Z, N_K*N_H*N_LOC)

    βZ_T = get_βZ_T(V_end_mat)
    mul!(V_preshock_mat, βZ_T, V_end_mat)

    permutedims!(V_preshock, V_preshock_perm, (2,1,3,4))
    return V_preshock
end

# Forward #
#---------#
function get_λ_postshock(λ_preshock, prealloc)
    (;λ_postshock_mat, λ_preshock_perm) = prealloc
    λ_postshock = prealloc.λ_end

    permutedims!(λ_preshock_perm, λ_preshock, (2,1,3,4))

    λ_preshock_mat = reshape(λ_preshock_perm, N_Z, N_K*N_H*N_LOC)
    z_t_transpose = get_Z_T_transpose(λ_preshock)
    mul!(λ_postshock_mat, z_t_transpose, λ_preshock_mat)

    λ_postshock_perm = reshape(λ_postshock_mat, N_Z, N_K, N_H, N_LOC)
    permutedims!(λ_postshock, λ_postshock_perm, (2,1,3,4))

    return λ_postshock
end
