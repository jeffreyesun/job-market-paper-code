
############################################
# Initial Distribution of Young Households #
############################################

const ONEHOT_RENTER = reshape(e_vec(N_H, 1), 1,1,N_H)
const ONEHOT_RENTER_GPU = gpu(ONEHOT_RENTER)
get_onehot_renter(::AbstractArray) = ONEHOT_RENTER
get_onehot_renter(::CuArray) = ONEHOT_RENTER_GPU

@inline function get_λ_born(young_loc, V_start)
    young_wealth = get_young_wealth_init(V_start)
    young_z_flat = get_young_z_flat(V_start)

    young_z = reshape(young_z_flat, 1, N_Z)

    young_h = get_onehot_renter(V_start)

    λ_born = @. young_wealth * young_z * young_h * young_loc
    λ_born ./= sum(λ_born; dims=1:ndims(λ_born))

    return λ_born .* POP_SUM ./ N_AGE
end

"Get initial distribution of young households, where young households enter to maximize V."
function get_λ_born_based_on_V(V_start, params)
    (;ψ) = params
    
    λ_born = get_λ_born(1, V_start)
    #V_start_rawlsian = weightedmean(V_start, λ_born; dims=(1,2,3))
    Vλ = V_start .* λ_born
    V_start_rawlsian = sum(Vλ; dims=(1,2,3)) ./ sum(λ_born; dims=(1,2,3))
    #V_mean = mean(V_start_rawlsian)
    V_mean = 23.0
    #NOTE Households don't observe their k and z before preference shocks and choosing

    eψV_start = @. exp(ψ*V_start_rawlsian - ψ*V_mean)
    young_loc = eψV_start ./ sum(eψV_start; dims=1:ndims(eψV_start))

    return get_λ_born(young_loc, V_start)
end

"""
Get initial distribution of young households, assuming the spatial distribution to be equal to
the contemporaneous spatial distribution of agei=4 (age 60-70) households.
"""
function get_λ_init_based_on_Λ(period_data::PeriodData, params::Params)
    (;λ_start) = period_data
    young_loc = sum(λ_start[4], dims=(1,2,3))
    return get_λ_init(young_loc)
end
