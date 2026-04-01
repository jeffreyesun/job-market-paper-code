
#############
# Constants #
#############

# State Space Grids #
const WEALTH_GRID_FLAT_GPU = gpu(WEALTH_GRID_FLAT)
const LOGZ_GRID_FLAT_GPU = gpu(LOGZ_GRID_FLAT)
const H_GRID_FLAT_GPU = gpu(H_GRID_FLAT)
const βZ_T_GPU = gpu(βZ_T)
const Z_T_TRANSPOSE_GPU = gpu(Z_T_TRANSPOSE)

# Shaped Grids for Broadcasting #
const WEALTH_GRID_GPU = gpu(WEALTH_GRID)
const WEALTH_NEXT_GRID_GPU = gpu(WEALTH_NEXT_GRID)
const LOGZ_GRID_GPU = gpu(LOGZ_GRID)
const Z_GRID_GPU = [gpu(Z_GRID[agei]) for agei=1:N_AGE]
const H_GRID_GPU = gpu(H_GRID)
const AGE_GRID_GPU = gpu(AGE_GRID)

# Other Constants #
const DIST_LOC_GPU = gpu(DIST_LOC)
const YOUNG_WEALTH_INIT_GPU = gpu(YOUNG_WEALTH_INIT)
const YOUNG_Z_FLAT_GPU = gpu(YOUNG_Z_FLAT)
const LOG_EXPENDITURE_MAT_GPU = gpu(LOG_EXPENDITURE_MAT)

# Getters to efficiently access constants depending on platform #
get_βZ_T(::AbstractArray) = βZ_T
get_βZ_T(::CuArray) = βZ_T_GPU
get_βZ_T_transpose(::AbstractArray) = Z_T_TRANSPOSE
get_βZ_T_transpose(::CuArray) = Z_T_TRANSPOSE_GPU
get_Z_T_transpose(::AbstractArray) = Z_T_TRANSPOSE
get_Z_T_transpose(::CuArray) = Z_T_TRANSPOSE_GPU

get_wealth_grid(::AbstractArray) = WEALTH_GRID
get_wealth_grid(::CuArray) = WEALTH_GRID_GPU
get_h_grid(::AbstractArray) = H_GRID
get_h_grid(::CuArray) = H_GRID_GPU
get_z_grid(agei, ::AbstractArray) = Z_GRID[agei]
get_z_grid(agei, ::CuArray) = Z_GRID_GPU[agei]

get_dist_loc(::AbstractArray) = DIST_LOC
get_dist_loc(::CuArray) = DIST_LOC_GPU
get_log_expenditure_mat(::AbstractArray) = LOG_EXPENDITURE_MAT
get_log_expenditure_mat(::CuArray) = LOG_EXPENDITURE_MAT_GPU
get_young_wealth_init(::AbstractArray) = YOUNG_WEALTH_INIT
get_young_wealth_init(::CuArray) = YOUNG_WEALTH_INIT_GPU
get_young_z_flat(::AbstractArray) = YOUNG_Z_FLAT
get_young_z_flat(::CuArray) = YOUNG_Z_FLAT_GPU

#########
# Types #
#########

# AgeSolution #
#-------------#
struct AgeSolutionGPU
    # Value Function
    ## End
    # Unshared V_end::Array{FLOAT, 4} = zeros(FLOAT, STATE_IDXs)
    ## Preshock
    V_end_perm::CuArray{FLOAT, 4, CUDA.DeviceMemory}
    V_preshock_perm::CuArray{FLOAT, 4, CUDA.DeviceMemory}
    V_preshock::CuArray{FLOAT, 4, CUDA.DeviceMemory}
    ## Consume
    V_consume::CuArray{FLOAT, 4, CUDA.DeviceMemory}
    ## Income
    # Unshared V_income::Array{FLOAT, 4} = zeros(FLOAT, STATE_IDXs)
    ## Sell
    # Unshared V_sell::Array{FLOAT, 4} = zeros(FLOAT, STATE_IDXs)
    # Unshared V_move::Array{FLOAT, 4} = zeros(FLOAT, STATE_IDXs)
    ## Buy
    # Unshared V_choosebuy::Array{FLOAT, 4} = zeros(FLOAT, STATE_IDXs)
    ## Start
    # Unshared V_start::Array{FLOAT, 4} = zeros(FLOAT, STATE_IDXs)
    # Population Distribution
    ## Move
    λ_move_postmove::CuArray{FLOAT, 4, CUDA.DeviceMemory}
    ## Consumption
    λ_postc::CuArray{FLOAT, 4, CUDA.DeviceMemory}
    ## Shock
    λ_preshock_perm::CuArray{FLOAT, 4, CUDA.DeviceMemory}
end
function to_gpu(as::AgeSolution)
    return AgeSolutionGPU(
        cu(as.V_end_perm),
        cu(as.V_preshock_perm),
        cu(as.V_preshock),
        cu(as.V_consume),
        cu(as.λ_move_postmove),
        cu(as.λ_postc),
        cu(as.λ_preshock_perm)
    )
end
function to_gpu!(asg::AgeSolutionGPU, as::AgeSolution)
    asg.V_end_perm .= cu(as.V_end_perm)
    asg.V_preshock_perm .= cu(as.V_preshock_perm)
    asg.V_preshock .= cu(as.V_preshock)
    asg.V_consume .= cu(as.V_consume)
    asg.λ_move_postmove .= cu(as.λ_move_postmove)
    asg.λ_postc .= cu(as.λ_postc)
    asg.λ_preshock_perm .= cu(as.λ_preshock_perm)
    return asg
end
function to_cpu!(as::AgeSolution, asg::AgeSolutionGPU)
    as.V_end_perm .= cpu(asg.V_end_perm)
    as.V_preshock_perm .= cpu(asg.V_preshock_perm)
    as.V_preshock .= cpu(asg.V_preshock)
    as.V_consume .= cpu(asg.V_consume)
    as.λ_move_postmove .= cpu(asg.λ_move_postmove)
    as.λ_postc .= cpu(asg.λ_postc)
    as.λ_preshock_perm .= cpu(asg.λ_preshock_perm)
    return as
end
function to_cpu(asg::AgeSolutionGPU)
    return to_cpu!(AgeSolution(), asg)
end
Base.Broadcast.broadcastable(asg::AgeSolutionGPU) = Ref(asg)

# PeriodSolutionSlice #
#---------------------#
struct PeriodSolutionSliceGPU{T}
    # Population Distribution
    ## Price
    # Unshared λ_start
    λ_postprice::CuArray{Float32, 4, CUDA.DeviceMemory}
    ## Sell
    λ_postsell::CuArray{Float32, 4, CUDA.DeviceMemory}
    ## Move
    λ_postmove::CuArray{Float32, 4, CUDA.DeviceMemory}
    ## Buy
    λ_postbuy::CuArray{Float32, 4, CUDA.DeviceMemory}
    ## Income
    # Unshared λ_postincome
    ## Consumption
    ## Shock
    λ_postshock_mat::CuArray{Float32, 2, CUDA.DeviceMemory}
    # Unshared λ_end
    ## Miscellaneous
    kloc_scratch::T
    origin_weights::CuArray{Float32, 4, CUDA.DeviceMemory}
    P_sell::CuArray{Float32, 4, CUDA.DeviceMemory}
    P_buy::CuArray{Float32, 4, CUDA.DeviceMemory}
    #^^^ Everything above used to be in AgeSolution
    # Value Function
    ## Next
    V_end::CuArray{Float32, 4, CUDA.DeviceMemory}
    ## Consume
    wealthi_postc_k_prec::CuArray{Int, 4, CUDA.DeviceMemory}
    ## Income
    V_income::CuArray{Float32, 4, CUDA.DeviceMemory}
    wealth_postinc_k_preinc::CuArray{Float32, 4, CUDA.DeviceMemory}
    ## Buy
    V_choosebuy::CuArray{Float32, 4, CUDA.DeviceMemory}
    ## Move
    eψV_move_tilde::CuArray{Float32, 4, CUDA.DeviceMemory}
    eψV_postmove_tilde::CuArray{Float32, 4, CUDA.DeviceMemory}
    ψV_means::CuArray{Float32, 4, CUDA.DeviceMemory}
    V_move::CuArray{Float32, 4, CUDA.DeviceMemory}
    ## Sell
    V_sell::CuArray{Float32, 4, CUDA.DeviceMemory}
    V_choosesell::CuArray{Float32, 4, CUDA.DeviceMemory}
    ## Start
    V_start::CuArray{Float32, 4, CUDA.DeviceMemory}
    
    # Population Distribution
    λ_start::CuArray{Float32, 4, CUDA.DeviceMemory}
    λ_postincome::CuArray{Float32, 4, CUDA.DeviceMemory}
    λ_end::CuArray{Float32, 4, CUDA.DeviceMemory}
end
function to_gpu(pss::PeriodSolutionSlice)
    return PeriodSolutionSliceGPU(
        cu(pss.λ_postprice),
        cu(pss.λ_postsell),
        cu(pss.λ_postmove),
        cu(pss.λ_postbuy),
        cu(pss.λ_postshock_mat),
        [cu(vec) for vec in pss.kloc_scratch],
        cu(pss.origin_weights),
        cu(pss.P_sell),
        cu(pss.P_buy),
        cu(pss.V_end),
        cu(pss.wealthi_postc_k_prec),
        cu(pss.V_income),
        cu(pss.wealth_postinc_k_preinc),
        cu(pss.V_choosebuy),
        cu(pss.eψV_move_tilde),
        cu(pss.eψV_postmove_tilde),
        cu(pss.ψV_means),
        cu(pss.V_move),
        cu(pss.V_sell),
        cu(pss.V_choosesell),
        cu(pss.V_start),
        cu(pss.λ_start),
        cu(pss.λ_postincome),
        cu(pss.λ_end)
    )
end
function to_gpu!(pssg::PeriodSolutionSliceGPU, pss::PeriodSolutionSlice)
    pssg.λ_postprice .= cu(pss.λ_postprice)
    pssg.λ_postsell .= cu(pss.λ_postsell)
    pssg.λ_postmove .= cu(pss.λ_postmove)
    pssg.λ_postbuy .= cu(pss.λ_postbuy)
    pssg.λ_postshock_mat .= cu(pss.λ_postshock_mat)
    for (vecg, vec) in zip(pssg.kloc_scratch, pss.kloc_scratch)
        vecg .= cu(vec)
    end
    pssg.origin_weights .= cu(pss.origin_weights)
    pssg.P_sell .= cu(pss.P_sell)
    pssg.P_buy .= cu(pss.P_buy)
    pssg.V_end .= cu(pss.V_end)
    pssg.wealthi_postc_k_prec .= cu(pss.wealthi_postc_k_prec)
    pssg.V_income .= cu(pss.V_income)
    pssg.wealth_postinc_k_preinc .= cu(pss.wealth_postinc_k_preinc)
    pssg.V_choosebuy .= cu(pss.V_choosebuy)
    pssg.eψV_move_tilde .= cu(pss.eψV_move_tilde)
    pssg.eψV_postmove_tilde .= cu(pss.eψV_postmove_tilde)
    pssg.ψV_means .= cu(pss.ψV_means)
    pssg.V_move .= cu(pss.V_move)
    pssg.V_sell .= cu(pss.V_sell)
    pssg.V_choosesell .= cu(pss.V_choosesell)
    pssg.V_start .= cu(pss.V_start)
    pssg.λ_start .= cu(pss.λ_start)
    pssg.λ_postincome .= cu(pss.λ_postincome)
    pssg.λ_end .= cu(pss.λ_end)
    return pssg
end
function to_cpu!(pss::PeriodSolutionSlice, pssg::PeriodSolutionSliceGPU)
    pss.λ_postprice .= cpu(pssg.λ_postprice)
    pss.λ_postsell .= cpu(pssg.λ_postsell)
    pss.λ_postmove .= cpu(pssg.λ_postmove)
    pss.λ_postbuy .= cpu(pssg.λ_postbuy)
    pss.λ_postshock_mat .= cpu(pssg.λ_postshock_mat)
    for (vec, vecg) in zip(pss.kloc_scratch, pssg.kloc_scratch)
        vec .= cpu(vecg)
    end
    pss.origin_weights .= cpu(pssg.origin_weights)
    pss.P_sell .= cpu(pssg.P_sell)
    pss.P_buy .= cpu(pssg.P_buy)
    pss.V_end .= cpu(pssg.V_end)
    pss.wealthi_postc_k_prec .= cpu(pssg.wealthi_postc_k_prec)
    pss.V_income .= cpu(pssg.V_income)
    pss.wealth_postinc_k_preinc .= cpu(pssg.wealth_postinc_k_preinc)
    pss.V_choosebuy .= cpu(pssg.V_choosebuy)
    pss.eψV_move_tilde .= cpu(pssg.eψV_move_tilde)
    pss.eψV_postmove_tilde .= cpu(pssg.eψV_postmove_tilde)
    pss.ψV_means .= cpu(pssg.ψV_means)
    pss.V_move .= cpu(pssg.V_move)
    pss.V_sell .= cpu(pssg.V_sell)
    pss.V_choosesell .= cpu(pssg.V_choosesell)
    pss.V_start .= cpu(pssg.V_start)
    pss.λ_start .= cpu(pssg.λ_start)
    pss.λ_postincome .= cpu(pssg.λ_postincome)
    pss.λ_end .= cpu(pssg.λ_end)
    return pss
end
function to_cpu(pssg::PeriodSolutionSliceGPU)
    return to_cpu!(PeriodSolutionSlice(), pssg)
end
Base.Broadcast.broadcastable(pssg::PeriodSolutionSliceGPU) = Ref(pssg)

# Precomputed #
#-------------#
struct PrecomputedGPU
    wealth_postprice_k_preprice::CuArray{Float32, 4, CUDA.DeviceMemory}
    wealth_postsell_k_presell::CuArray{Float32, 4, CUDA.DeviceMemory}
    
    κqh_own::CuArray{Float32, 4, CUDA.DeviceMemory}
    forbidden_states::CuArray{Float32, 4, CUDA.DeviceMemory}
    
    eψFu_inv::CuArray{Float32, 2, CUDA.DeviceMemory}
    u_indirect_loc::CuVector{Float32, CUDA.DeviceMemory}
end
function to_gpu(precomp::Precomputed)
    return PrecomputedGPU(
        cu(precomp.wealth_postprice_k_preprice),
        cu(precomp.wealth_postsell_k_presell),
        cu(precomp.κqh_own),
        cu(precomp.forbidden_states),
        cu(precomp.eψFu_inv),
        cu(precomp.u_indirect_loc)
    )
end
function to_gpu!(precompg::PrecomputedGPU, precomp::Precomputed)
    precompg.wealth_postprice_k_preprice .= cu(precomp.wealth_postprice_k_preprice)
    precompg.wealth_postsell_k_presell .= cu(precomp.wealth_postsell_k_presell)
    precompg.κqh_own .= cu(precomp.κqh_own)
    precompg.forbidden_states .= cu(precomp.forbidden_states)
    precompg.eψFu_inv .= cu(precomp.eψFu_inv)
    precompg.u_indirect_loc .= cu(precomp.u_indirect_loc)
    return precompg
end
function to_cpu!(precomp::Precomputed, precompg::PrecomputedGPU)
    precomp.wealth_postprice_k_preprice .= cpu(precompg.wealth_postprice_k_preprice)
    precomp.wealth_postsell_k_presell .= cpu(precompg.wealth_postsell_k_presell)
    precomp.κqh_own .= cpu(precompg.κqh_own)
    precomp.forbidden_states .= cpu(precompg.forbidden_states)
    precomp.eψFu_inv .= cpu(precompg.eψFu_inv)
    precomp.u_indirect_loc .= cpu(precompg.u_indirect_loc)
    return precomp
end
function to_cpu(precompg::PrecomputedGPU)
    return to_cpu!(Precomputed(), precompg)
end
Base.Broadcast.broadcastable(precompg::PrecomputedGPU) = Ref(precompg)

# Prices #
#--------#
struct PricesGPU
    q::CuArray{Float32, 4, CUDA.DeviceMemory}
    ρ::CuArray{Float32, 4, CUDA.DeviceMemory}
    q_last::CuArray{Float32, 4, CUDA.DeviceMemory}
end
function to_gpu(prices::Prices)
    return PricesGPU(
        cu(prices.q),
        cu(prices.ρ),
        cu(prices.q_last)
    )
end
function to_gpu!(pricesg::PricesGPU, prices::Prices)
    pricesg.q .= cu(prices.q)
    pricesg.ρ .= cu(prices.ρ)
    pricesg.q_last .= cu(prices.q_last)
    return pricesg
end
function to_cpu!(prices::Prices, pricesg::PricesGPU)
    prices.q .= cpu(pricesg.q)
    prices.ρ .= cpu(pricesg.ρ)
    prices.q_last .= cpu(pricesg.q_last)
    return prices
end
to_cpu(pricesg::PricesGPU) = to_cpu!(Prices(N_LOC), pricesg)
Base.Broadcast.broadcastable(pricesg::PricesGPU) = Ref(pricesg)

# AgeData #
#---------#
struct AgeDataGPU{T}
    agei::Int
    age_solution::AgeSolutionGPU
    period_solution_slice::PeriodSolutionSliceGPU{T}
    precomputed::PrecomputedGPU
    prices::PricesGPU
end
function to_gpu(ad::AgeData)
    return AgeDataGPU(
        ad.agei,
        to_gpu(ad.age_solution),
        to_gpu(ad.period_solution_slice),
        to_gpu(ad.precomputed),
        to_gpu(ad.prices)
    )
end
function to_gpu!(adg::AgeDataGPU, ad::AgeData)
    adg.agei = ad.agei
    to_gpu!(adg.age_solution, ad.age_solution)
    to_gpu!(adg.period_solution_slice, ad.period_solution_slice)
    to_gpu!(adg.precomputed, ad.precomputed)
    to_gpu!(adg.prices, ad.prices)
    return adg
end
function to_cpu(adg::AgeDataGPU)
    return AgeData(
        adg.agei,
        to_cpu(adg.age_solution),
        to_cpu(adg.period_solution_slice),
        to_cpu(adg.precomputed),
        to_cpu(adg.prices)
    )
end
function to_cpu!(ad::AgeData, adg::AgeDataGPU)
    to_cpu!(ad.age_solution, adg.age_solution)
    to_cpu!(ad.period_solution_slice, adg.period_solution_slice)
    to_cpu!(ad.precomputed, adg.precomputed)
    to_cpu!(ad.prices, adg.prices)
    return ad
end
Base.Broadcast.broadcastable(ad::AgeDataGPU) = Ref(ad)

# PeriodSolution #
#----------------#
struct PeriodSolutionGPU{T}
    period_solution_slices::Vector{PeriodSolutionSliceGPU{T}}
end
Base.getindex(ps::PeriodSolutionGPU, i::Int) = ps.period_solution_slices[i]
function to_gpu(ps::PeriodSolution)
    return PeriodSolutionGPU([to_gpu(ps.period_solution_slices[agei]) for agei=1:N_AGE])
end
function to_gpu!(pssg::PeriodSolutionGPU, ps::PeriodSolution)
    to_gpu!.(pssg.period_solution_slices, ps.period_solution_slices)
    return pssg
end
function to_cpu!(ps::PeriodSolution, pssg::PeriodSolutionGPU)
    to_cpu!.(ps.period_solution_slices, pssg.period_solution_slices)
    return ps
end
to_cpu(pssg::PeriodSolutionGPU) = to_cpu!(PeriodSolution(), pssg)
Base.Broadcast.broadcastable(pssg::PeriodSolutionGPU) = Ref(pssg)

# PeriodData #
#------------#
struct PeriodDataGPU{T}
    age_solution::AgeSolutionGPU
    period_solution::PeriodSolutionGPU{T}
    precomputed::PrecomputedGPU
    prices::PricesGPU
end
function to_gpu(pd::PeriodData)
    return PeriodDataGPU(
        to_gpu(pd.age_solution),
        to_gpu(pd.period_solution),
        to_gpu(pd.precomputed),
        to_gpu(pd.prices)
    )
end
to_gpu(pd_g::PeriodDataGPU) = pd_g
function to_gpu!(pd_g::PeriodDataGPU, pd::PeriodData)
    to_gpu!(pd_g.age_solution, pd.age_solution)
    to_gpu!(pd_g.period_solution, pd.period_solution)
    to_gpu!(pd_g.precomputed, pd.precomputed)
    to_gpu!(pd_g.prices, pd.prices)
    return pd_g
end
function to_cpu!(pd::PeriodData, pdg::PeriodDataGPU)
    to_cpu!(pd.age_solution, pdg.age_solution)
    to_cpu!(pd.period_solution, pdg.period_solution)
    to_cpu!(pd.precomputed, pdg.precomputed)
    to_cpu!(pd.prices, pdg.prices)
    return pd
end
function to_cpu!(pd_g::PeriodDataGPU, pd_g2::PeriodDataGPU)
    if pd_g === pd_g2
        return pd_g
    else
        @error "Copying from one PeriodDataGPU to another PeriodDataGPU not yet implemented."
    end
end
to_cpu(pdg::PeriodDataGPU) = to_cpu!(PeriodData(), pdg)
Base.Broadcast.broadcastable(pd::PeriodDataGPU) = Ref(pd)


#############
# Accessors #
#############

"Any subfield of AgeDataGPU can be accessed as a property."
function Base.getproperty(ad::AgeDataGPU, s::Symbol)
    if hasfield(AgeDataGPU, s)
        return getfield(ad, s)
    elseif hasfield(AgeSolutionGPU, s)
        return getfield(getfield(ad, :age_solution), s)
    elseif hasfield(PeriodSolutionSliceGPU, s)
        return getfield(getfield(ad, :period_solution_slice), s)
    elseif hasfield(PrecomputedGPU, s)
        return getfield(getfield(ad, :precomputed), s)
    else
        return getfield(getfield(ad, :prices), s)
    end
end

"You can access properties of PeriodSolutionSlice on PeriodSolution, receiving a vector by age."
function Base.getproperty(ps::PeriodSolutionGPU, s::Symbol)
    if s === :period_solution_slices
        return getfield(ps, :period_solution_slices)
    else
        return getfield.(ps.period_solution_slices, s)
    end
end

"Any subfield of PeriodData can be accessed as a property."
function Base.getproperty(pd::PeriodDataGPU, s::Symbol)
    if hasfield(PeriodDataGPU, s)
        return getfield(pd, s)
    elseif hasfield(AgeSolutionGPU, s)
        return getfield(getfield(pd, :age_solution), s)
    elseif hasfield(PeriodSolutionSliceGPU, s)
        return getproperty(getfield(pd, :period_solution), s)
    elseif hasfield(PrecomputedGPU, s)
        return getfield(getfield(pd, :precomputed), s)
    else
        return getproperty(getfield(pd, :prices), s)
    end
end

function AgeDataGPU(period_data::PeriodDataGPU, agei::Int)
    (;age_solution, period_solution, precomputed, prices) = period_data
    return AgeDataGPU(agei, age_solution, period_solution[agei], precomputed, prices)
end
Base.getindex(pd::PeriodDataGPU, agei::Int) = AgeDataGPU(pd, agei)
Base.lastindex(pd::PeriodDataGPU) = length(pd.period_solution.period_solution_slices)


##########
# Params #
##########

struct SpatialGPU
    GISMATCH::CuArray{Int64, 4, CUDA.DeviceMemory}

    # Current Conditions #
    A::CuArray{Float32, 4, CUDA.DeviceMemory}
    α::CuArray{Float32, 4, CUDA.DeviceMemory}
    δ::CuArray{Float32, 4, CUDA.DeviceMemory}
    H_S::CuArray{Float32, 4, CUDA.DeviceMemory}

    # Fundamentals #
    # Housing
    elasticity::CuArray{Float32, 4, CUDA.DeviceMemory}
    Π::CuArray{Float32, 4, CUDA.DeviceMemory}

    # Climate
    ## Intercept
    A_bar::CuArray{Float32, 4, CUDA.DeviceMemory}
    α_bar::CuArray{Float32, 4, CUDA.DeviceMemory}
    δ_bar::CuArray{Float32, 4, CUDA.DeviceMemory}
    ## Slope (per °C global warming)
    A_g::CuArray{Float32, 4, CUDA.DeviceMemory}
    α_g::CuArray{Float32, 4, CUDA.DeviceMemory}
    δ_g::CuArray{Float32, 4, CUDA.DeviceMemory}
end
function to_gpu(spatial::StructArray{Location, 4})
    return SpatialGPU(
        cu(spatial.GISMATCH),
        cu(spatial.A),
        cu(spatial.α),
        cu(spatial.δ),
        cu(spatial.H_S),
        cu(spatial.elasticity),
        cu(spatial.Π),
        cu(spatial.A_bar),
        cu(spatial.α_bar),
        cu(spatial.δ_bar),
        cu(spatial.A_g),
        cu(spatial.α_g),
        cu(spatial.δ_g)
    )
end
to_gpu(spatialg::SpatialGPU) = spatialg
function to_cpu!(spatial, spatialg::SpatialGPU)
    spatial.GISMATCH .= cpu(spatialg.GISMATCH)
    spatial.A .= cpu(spatialg.A)
    spatial.α .= cpu(spatialg.α)
    spatial.δ .= cpu(spatialg.δ)
    spatial.H_S .= cpu(spatialg.H_S)
    spatial.elasticity .= cpu(spatialg.elasticity)
    spatial.Π .= cpu(spatialg.Π)
    spatial.A_bar .= cpu(spatialg.A_bar)
    spatial.α_bar .= cpu(spatialg.α_bar)
    spatial.δ_bar .= cpu(spatialg.δ_bar)
    spatial.A_g .= cpu(spatialg.A_g)
    spatial.α_g .= cpu(spatialg.α_g)
    spatial.δ_g .= cpu(spatialg.δ_g)
    return spatial
end
to_cpu(spatialg::SpatialGPU) = to_cpu!(StructArray{Location, 4}(N_LOC), spatialg)

function to_gpu(params::Params)
    (;γ, σ, ϕ, κ, ξ, F_u_fixed, F_u_dist, χ, r_m, bequest_motive, ψ, δ_dep, decade, scenario_ind) = params
    return Params(;γ, σ, ϕ, κ, ξ, F_u_fixed, F_u_dist, χ, r_m, bequest_motive, ψ, δ_dep, decade, scenario_ind, spatial=to_gpu(params.spatial))
end
function to_cpu(paramsg::Params)
    (;γ, σ, ϕ, κ, ξ, F_u_fixed, F_u_dist, χ, r_m, bequest_motive, ψ, δ_dep, decade, scenario_ind) = paramsg
    return Params(;γ, σ, ϕ, κ, ξ, F_u_fixed, F_u_dist, χ, r_m, bequest_motive, ψ, δ_dep, decade, scenario_ind, spatial=to_cpu(paramsg.spatial))
end
function to_cpu!(params::Params, paramsg::Params)
    params.γ = paramsg.γ
    params.σ = paramsg.σ
    params.ϕ = paramsg.ϕ
    params.κ = paramsg.κ
    params.ξ = paramsg.ξ
    params.F_u_fixed = paramsg.F_u_fixed
    params.F_u_dist = paramsg.F_u_dist
    params.χ = paramsg.χ
    params.r_m = paramsg.r_m
    params.bequest_motive = paramsg.bequest_motive
    params.ψ = paramsg.ψ
    params.δ_dep = paramsg.δ_dep
    params.decade = paramsg.decade
    params.scenario_ind = paramsg.scenario_ind
    to_cpu!(params.spatial, paramsg.spatial)
    return params
end


################
# StructArrays #
################

function to_gpu(sa::StructArray)
    allowscalar(true)
    sa_gpu = cu(sa)
    allowscalar(false)
    return sa_gpu
end
