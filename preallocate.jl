
using Accessors

################
# Data Structs #
################

"""
Data for a single age group, intended to be reused for each age group calculation,
so that these values cannot be relied upon after the entire period has been solved.
"""
@kwdef struct AgeSolution{T<:Number}
    # Value Function
    ## Next
    # Unshared V_next::Array{T, 5} = zeros(T, STATE_IDXs)
    ## Preshock
    V_next_perm::Array{T, 5} = zeros(T, N_Z, N_K, N_Hli, N_Hle, N_LOC)
    V_preshock_perm::Array{T, 5} = zeros(T, N_Z, N_K, N_Hli, N_Hle, N_LOC)
    V_preshock::Array{T, 5} = zeros(T, STATE_IDXs)
    ## Consume
    V_consume::Array{T, 5} = zeros(T, STATE_IDXs)
    ## Income
    # Unshared V_income::Array{T, 5} = zeros(T, STATE_IDXs)
    ## Market
    V_choosebuy_let::Array{T, 5} = zeros(T, (N_K,N_Z,N_Hli,1,N_LOC))
    # Unshared V_sell_let::Array{T, 5} = zeros(T, STATE_IDXs)
    # Unshared V_choosesell_let::Array{T, 5} = zeros(T, STATE_IDXs)
    V_choosebuy_live::Array{T, 5} = zeros(T, (N_K,N_Z,1,N_Hle,N_LOC))
    # Unshared V_sell_live::Array{T, 5} = zeros(T, STATE_IDXs)
    V_choosesell_live::Array{T, 5} = zeros(T, STATE_IDXs)
    ## Move
    V_move_k_postmove::Array{T, 5} = zeros(T, N_K, N_Z, 1, 1, N_LOC)
    eψV_move_tilde::Array{T, 5} = zeros(T, STATE_IDXs)
    V_move::Array{T, 5} = zeros(T, STATE_IDXs)
    ## Choose Move
    V_choosemove::Array{T, 5} = zeros(T, STATE_IDXs)
    ## Start
    eψV_nomove_tilde::Array{T, 5} = zeros(T, STATE_IDXs)
    # Unshared V_price::Array{T,5} = zeros(T, STATE_IDXs)

    # Population Distribution
    # Unshared λ_start
    λ_postprice::Array{T, 5} = zeros(T, STATE_IDXs)
    λ_move::Array{T, 5} = zeros(T, STATE_IDXs)
    λ_nomove::Array{T, 5} = zeros(T, STATE_IDXs)
    λ_move_k_postmove::Array{T, 5} = zeros(T, N_K, N_Z, 1, 1, N_LOC)
    λ_postmove::Array{T, 5} = zeros(T, N_K, N_Z, 1, 1, N_LOC)
    λ_premarket::Array{T, 5} = zeros(T, STATE_IDXs)
    λ_prebuy_live::Array{T, 5} = zeros(T, STATE_IDXs)
    λ_presell_let::Array{T, 5} = zeros(T, STATE_IDXs)
    λ_prebuy_let::Array{T, 5} = zeros(T, STATE_IDXs)
    λ_postmarket::Array{T, 5} = zeros(T, STATE_IDXs)
    # Unshared λ_prec
    λ_preshock::Array{T, 5} = zeros(T, STATE_IDXs)
    λ_next::Array{T, 5} = zeros(T, STATE_IDXs)
    λ_postmarket_perm::Array{T, 5} = zeros(T, N_Z, N_K, N_Hli, N_Hle, N_LOC)
    λ_next_mat::Array{T, 2} = zeros(T, N_Z, N_K*N_Hli*N_Hle*N_LOC)

    # Miscellaneous
    kloc_scratch::Vector{Array{T, 1}} = [zeros(T, N_K) for _ in 1:N_LOC]
    origin_weights::Array{T, 5} = zeros(T, N_K, N_Z, 1, 1, N_LOC)
    P_sell_live::Array{T, 5} = zeros(T, STATE_IDXs)
    P_buy_live::Array{T, 5} = zeros(T, STATE_IDXs)
    P_sell_let::Array{T, 5} = zeros(T, STATE_IDXs)
    P_buy_let::Array{T, 5} = zeros(T, STATE_IDXs)
end
AgeSolution(; kwargs...) = AgeSolution{FLOAT_PRECISION}(; kwargs...)

"""
Data for a single age group slice, intended to be separately allocated for each
age group, so that this *can* be relied upon to represent an age group, even
after other age groups have been solved.
"""
@kwdef struct PeriodSolutionSlice{T<:Number}
    # Value Function
    ## Next
    V_next::Array{T, 5} = zeros(T, STATE_IDXs)
    ## Consume
    wealthi_postc_k_prec::Array{Int,5} = zeros(Int, STATE_IDXs)
    ## Income
    V_income::Array{T, 5} = zeros(T, STATE_IDXs)
    wealth_postinc_k_preinc::Array{T,5} = zeros(T, STATE_IDXs)
    ## Market
    V_sell_let::Array{T, 5} = zeros(T, STATE_IDXs)
    V_choosesell_let::Array{T, 5} = zeros(T, STATE_IDXs)
    V_sell_live::Array{T, 5} = zeros(T, STATE_IDXs)
    ## Postmove
    eψV_postmove_tilde::Array{T,5} = zeros(T, N_K, N_Z, 1, 1, N_LOC)
    ## Move
    ψV_means::Array{T,5} = zeros(T, N_K, N_Z, 1, 1, 1)
    eψV_move_k_postmove_tilde::Array{T,5} = zeros(T, N_K, N_Z, 1, 1, N_LOC)
    P_move::Array{T,5} = zeros(T, STATE_IDXs)
    ## Start
    V_price::Array{T,5} = zeros(T, STATE_IDXs)
    
    # Population Distribution
    λ_start::Array{T,5} = zeros(T, STATE_IDXs)
    λ_prec::Array{T,5} = zeros(T, STATE_IDXs)
end
PeriodSolutionSlice(; kwargs...) = PeriodSolutionSlice{FLOAT_PRECISION}(; kwargs...)

"Precomputed data, for a given set of parameters."
@kwdef struct Precomputed{T<:Number}
    wealth_postprice_k_preprice::Array{T,5} = zeros(T, N_K, 1, N_Hli, N_Hle, N_LOC)
    wealth_postmove_k_presell::Array{T,5} = zeros(T, N_K, 1, N_Hli, N_Hle, N_LOC)
    wealth_postsell_live_k_presell::Array{T,5} = zeros(T, N_K, 1, N_Hli, 1, N_LOC)
    wealth_postsell_let_k_presell::Array{T,5} = zeros(T, N_K, 1, 1, N_Hle, N_LOC)
    wealth_postsell_both_k_presell::Array{T,5} = zeros(T, N_K, 1, N_Hli, N_Hle, N_LOC)
    
    κqh_own::Array{T,5} = zeros(T, 1, 1, N_Hli, N_Hle, N_LOC)
    forbidden_states::Array{T,5} = zeros(T, N_K, 1, N_Hli, N_Hle, N_LOC)
    u_indirect_rent::Vector{Vector{Vector{T}}} = [[zeros(T, N_K) for _ in 1:N_K] for _ in 1:N_LOC]
    u_indirect_own::Matrix{Vector{Vector{T}}} = [[zeros(T, N_K) for _ in 1:N_K] for _ in CartesianIndices((N_Hli,N_LOC))]
    
    eψFu_inv::Array{T,2} = zeros(T, N_LOC, N_LOC)
end
Precomputed(; kwargs...) = Precomputed{FLOAT_PRECISION}(; kwargs...)


##########################
# Data Struct Containers #
##########################

# By Age Group #
#--------------#
"""
A struct containing all the data that a particular age group slice needs to make their decisions.
"""
struct AgeData{T}
    agei::Int
    age_solution::AgeSolution{T}
    period_solution_slice::PeriodSolutionSlice{T}
    precomputed::Precomputed{T}
    loc_grid::LocGrid{T} # Not a concrete type!
end
AgeData(args...) = AgeData{FLOAT_PRECISION}(args...)

# By Period #
#-----------#
struct PeriodSolution{T}
    period_solution_slices::Vector{PeriodSolutionSlice{T}}
end
PeriodSolution{T}() where T = PeriodSolution{T}([PeriodSolutionSlice() for _ in 1:N_AGE])
PeriodSolution(period_solution_slices) = PeriodSolution{FLOAT_PRECISION}(period_solution_slices)
PeriodSolution() = PeriodSolution{FLOAT_PRECISION}()
Base.getindex(ps::PeriodSolution, i::Int) = ps.period_solution_slices[i]

@kwdef struct PeriodData{T}
    age_solution::AgeSolution{T} = AgeSolution()
    period_solution::PeriodSolution{T} = PeriodSolution()
    precomputed::Precomputed{T} = Precomputed()
    loc_grid::LocGrid{T} = LocGrid()
end
PeriodData(args...) = PeriodData{FLOAT_PRECISION}(args...)

# By Transition Path #
#--------------------#
struct PathData{T}
    n_dec::Int
    #age_solution_path::Vector{AgeSolution{T}}
    age_solution::AgeSolution{T}
    period_solution_path::Vector{PeriodSolution{T}}
    precomputed_path::Vector{Precomputed{T}}
    loc_grid_path::LocGridPath{T}
end

PeriodDataPath{T} = Vector{PeriodData{T}}

# Subfield Accessors #
#--------------------#
"Any subfield of AgeData can be accessed as a property."
function Base.getproperty(ad::AgeData, s::Symbol)
    if hasfield(AgeData, s)
        return getfield(ad, s)
    elseif hasfield(AgeSolution, s)
        return getfield(getfield(ad, :age_solution), s)
    elseif hasfield(PeriodSolutionSlice, s)
        return getfield(getfield(ad, :period_solution_slice), s)
    else hasfield(Precomputed, s)
        return getfield(getfield(ad, :precomputed), s)
    end
end

"You can access properties of PeriodSolutionSlice on PeriodSolution, receiving a vector by age."
function Base.getproperty(ps::PeriodSolution, s::Symbol)
    if s === :period_solution_slices
        return getfield(ps, :period_solution_slices)
    else
        return getfield.(ps.period_solution_slices, s)
    end
end

"You can access properties of PeriodSolutionSlice on PeriodSolutionPath, receiving a matrix by age and decade."
function get_periodsolutionslice_mat(psp::Vector{<:PeriodSolution}, s::Symbol)
    n_dec = length(psp)
    arrs = [getproperty(psp[deci][agei], s) for agei=1:N_AGE, deci=1:n_dec]
    return reshape(arrs, (1,1,1,1,1,N_AGE,1,n_dec))
end

"Any subfield of PeriodData can be accessed as a property."
function Base.getproperty(pd::PeriodData, s::Symbol)
    if hasfield(PeriodData, s)
        return getfield(pd, s)
    elseif hasfield(AgeSolution, s)
        return getfield(getfield(pd, :age_solution), s)
    elseif hasfield(PeriodSolutionSlice, s)
        return getproperty(getfield(pd, :period_solution), s)
    elseif hasfield(Precomputed, s)
        return getfield(getfield(pd, :precomputed), s)
    else
        return getproperty(getfield(pd, :loc_grid), s)
    end
end

"Any subfield of PathData can be accessed as a property."
function Base.getproperty(pd::PathData, s::Symbol)
    if hasfield(PathData, s)
        return getfield(pd, s)
    elseif hasfield(AgeSolution, s)
        return getfield(getfield(pd, :age_solution), s)
    elseif hasfield(PeriodSolutionSlice, s)
        return get_periodsolutionslice_mat(getfield(pd, :period_solution_path), s)
    elseif hasfield(Precomputed, s)
        return reshape(getfield.(getfield(pd, :precomputed_path), s), (1,1,1,1,1,1,1,pd.n_dec))
    else
        return getproperty(getfield(pd, :loc_grid_path), s)
    end
end

# Constructors #
#--------------#

get_X_agei(loc_grid::LocGrid, agei::Int) = (X_GRID[agei]..., loc_grid)
get_X_agei(pd::PeriodData, agei::Int) = get_X_agei(pd.loc_grid, agei)

"Preallocate a PathData struct."
function PathData(loc_grid_path::LocGridPath{T}) where T
    n_dec = size(loc_grid_path, DEC_DIM)
    #age_solution_path = [AgeSolution{T}() for _ in 1:n_dec]
    age_solution = AgeSolution{T}()
    period_solution_path = [PeriodSolution{T}() for _ in 1:n_dec]
    precomputed_path = [Precomputed{T}() for _ in 1:n_dec]
    #return PathData{T}(n_dec, age_solution_path, period_solution_path, precomputed_path, loc_grid_path)
    return PathData{T}(n_dec, age_solution, period_solution_path, precomputed_path, loc_grid_path)
end

"Extract a single decade slice from a LocGridPath."
LocGrid(loc_grid_path::LocGridPath, deci::Int) = reshape(selectdim(loc_grid_path, DEC_DIM, deci), (1,1,1,1,N_LOC))
get_n_dec(loc_grid_path::LocGridPath) = size(loc_grid_path, DEC_DIM)

"Extract all decade slices from a LocGridPath as a vector."
get_loc_grid_vec(lgp::LocGridPath) = LocGrid.(Ref(lgp), 1:size(lgp, DEC_DIM))

"Stack a bunch of LocGrids (or other similar arrays) into one big array."
function stack_locdata(locdata::Vector{<:AbstractArray})
    return stack(reshape.(locdata, Ref((1,1,1,1,N_LOC,1,1))); dims=DEC_DIM)
end
LocGridPath(loc_grid_vec::Vector{<:LocGrid}) = stack_locdata(loc_grid_vec)

"Extract a single decade slice from a PathData."
function PeriodData(path_data::PathData, deci::Int)
    #(;age_solution_path, period_solution_path, precomputed_path, loc_grid_path) = path_data
    #return PeriodData(age_solution_path[deci], period_solution_path[deci], precomputed_path[deci], LocGrid(loc_grid_path, deci))
    (;age_solution, period_solution_path, precomputed_path, loc_grid_path) = path_data
    return PeriodData(age_solution, period_solution_path[deci], precomputed_path[deci], LocGrid(loc_grid_path, deci))
end

"Extract a single decade-age slice from a PathData."
function AgeData(path_data::PathData, agei::Int, deci::Int)#; age_solution_deci::Int=deci)
    #(;age_solution_path, period_solution_path, precomputed_path, loc_grid_path) = path_data
    (;age_solution, period_solution_path, precomputed_path, loc_grid_path) = path_data
    period_solution_slice = period_solution_path[deci][agei]
    #age_solution = age_solution_path[age_solution_deci]
    precomputed = precomputed_path[deci]
    loc_grid = LocGrid(loc_grid_path, deci)
    return AgeData(agei, age_solution, period_solution_slice, precomputed, loc_grid)
end

"""
Return views into the slice at agei of each field of PeriodData, as an AgeData object.
Slice and combine the preallocated data to get all necessary data for a single age slice.
"""
function AgeData(period_data::PeriodData, agei::Int)
    (;age_solution, period_solution, precomputed, loc_grid) = period_data
    return AgeData(agei, age_solution, period_solution[agei], precomputed, loc_grid)
end

PeriodDataPath(path_data::PathData) = [PeriodData(path_data, deci) for deci=1:path_data.n_dec]

##############
# Precompute #
##############

"Precompute data for a given set of prices and parameters."
function precompute!(precomp::Precomputed, loc_grid::LocGrid, params::Params; new_params=true)
    (;γ, ψ, F_u_fixed, F_u_dist) = params
    (;eψFu_inv, κqh_own, forbidden_states,
        u_indirect_rent, u_indirect_own,
        wealth_postprice_k_preprice,
        wealth_postmove_k_presell, wealth_postsell_live_k_presell,
        wealth_postsell_let_k_presell, wealth_postsell_both_k_presell,
    ) = precomp
    
    if new_params
        @. eψFu_inv = inv(fastexp(ψ*(F_u_fixed + F_u_dist*DIST_LOC)))
        # eFu_inv represents the attractiveness of a destination relative to an origin
        # [origin, destination]
        # higher values are better
        #NOTE: Delete this line to allow within-location moving:
        eψFu_inv .*= 1 .- I(N_LOC)
    end

    # Note: this κ is 1-κ in the paper
    @. κqh_own = getκqh_own(H_LIVE_GRID, H_LET_GRID, loc_grid, params)
    @. forbidden_states = -Inf*(WEALTH_GRID < κqh_own)
    
    u_indirect_rent_big = get_indirect_u(loc_grid, params)
    u_indirect_own_big = get_u_own(loc_grid, params)

    Threads.@threads for ki=1:N_K
        for loci=1:N_LOC
            u_indirect_rent[loci][ki] .= @view(u_indirect_rent_big[ki,1,1,1,loci,1,:])

            for hlii=1:N_Hli
                u_indirect_own[hlii,loci][ki] .= @view(u_indirect_own_big[ki,1,hlii,1,loci,1,:])
            end
        end
    end

    # Compute wealth effects of price change
    @. wealth_postprice_k_preprice = get_wealth_postprice(WEALTH_GRID, H_LIVE_GRID, H_LET_GRID, loc_grid)
    # Compute wealth effects of realtor fees, mobility costs
    @. wealth_postmove_k_presell = get_wealth_postmove_from_presell(WEALTH_GRID, H_LIVE_GRID, H_LET_GRID, loc_grid, params)
    # wealth_postsell, in terms of wealth_presell
    @. wealth_postsell_live_k_presell = get_wealth_postsell(WEALTH_GRID, H_LIVE_GRID, loc_grid, params)
    @. wealth_postsell_let_k_presell  = get_wealth_postsell(WEALTH_GRID, H_LET_GRID,  loc_grid, params)

    return precomp
end


############################
# Update Preallocated Data #
############################

# Precompute #
#------------#
function precompute!(path_data::PathData, params::Params)
    (;precomputed_path, loc_grid_path) = path_data
    for (deci, precomputed) in enumerate(precomputed_path)
        precompute!(precomputed, LocGrid(loc_grid_path, deci), params)
    end
    return path_data
end

function precompute!(period_data::PeriodData, params::Params)
    (;precomputed, loc_grid) = period_data
    precompute!(precomputed, loc_grid, params)
    return period_data
end

# Apply Loc Grid #
#----------------#
function apply_loc_grid_path!(path_data::PathData, loc_grid_path::LocGridPath, params::Params, precompute=true)
    path_data.loc_grid_path .= loc_grid_path
    precompute && precompute!(path_data, params)
    return path_data
end

function apply_loc_grid!(period_data::PeriodData, loc_grid::LocGrid, params::Params, precompute=true)
    period_data.loc_grid .= loc_grid
    precompute && precompute!(period_data, params)
    return period_data
end

# Apply Climate #
#---------------#
function apply_climate_to_loc_grid!(path_data::PathData, climpath, params::Params, precompute=true)
    loc_grid_path = get_loc_climate.(path_data.loc_grid_path, climpath.ΔSST)
    return apply_loc_grid_path!(path_data, loc_grid_path, params, precompute)
end

function apply_climate_to_loc_grid!(period_data::PeriodData, ΔSST, params::Params, precompute=true)
    loc_grid = get_loc_climate.(period_data.loc_grid, ΔSST)
    return apply_loc_grid!(period_data, loc_grid, params, precompute)
end

# Preallocate and Precompute #
#----------------------------#
function preallocate_and_precompute(loc_grid::LocGrid, params::Params)
    return precompute!(PeriodData(;loc_grid), params)
end

function preallocate_and_precompute(loc_grid_path::LocGridPath, params::Params)
    path_data = PathData(loc_grid_path)
    precompute!(path_data, params)
    return path_data
end
