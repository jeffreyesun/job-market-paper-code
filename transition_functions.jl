
"""
Functions to invert λ, see where people are coming from.
"""

"""
Get the index of the presell state that corresponds to the given
premarket state for a household who does not move.
"""
function invert_λ_premarket(premarket_idx, ad::AgeData)
    (;λ_nomove, sell_live, sell_let) = ad

    zi = premarket_idx[Z_DIM]
    loci = premarket_idx[LOC_DIM]
    possible_presell_idxs = CartesianIndices((N_K, zi:zi, N_Hli, N_Hle, loci:loci))
    return filter(possible_presell_idxs) do idx
        get_premarket_idx(λ_nomove, sell_live, sell_let, idx)==premarket_idx
    end
end

function get_λ_preshock(λ_prec, sim_prealloc)
    (;wealthi_postc_k_prec, λ_preshock) = sim_prealloc

    λ_preshock .= 0

    @inbounds Threads.@threads for loci=1:N_LOC
        for idx=CartesianIndices((N_K, N_Z, N_Hli, N_Hle,loci:loci))
            for ki=1:N_K
                k1i = wealthi_postc_k_prec[ki,idx]
                λ_preshock[k1i,idx] += λ_prec[ki,idx]
            end
            # @views λ_preshock[[wealthi_postc_k_prec[:,idx],idx]] .+= λ_prec[:,idx]
        end
    end
    
    return λ_preshock
end

function invert_λ_preshock(preshock_idx, ad::AgeData)
    (;wealthi_postc_k_prec) = ad
    fixed_idx = CartesianIndex(Tuple(preshock_idx)[2:end])
    k1_by_k = @view wealthi_postc_k_prec[:,fixed_idx]
    kis_prec = findall(==(preshock_idx[K_DIM]), k1_by_k)
    return CartesianIndex.(kis_prec, Ref(fixed_idx))
end
