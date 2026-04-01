
#using Accessors

"""
Data for a single age group, intended to be reused for each age group calculation,
so that these values cannot be relied upon after the entire period has been solved.
This should be treated as a cache only, and not as persistent data in any way.
"""
@kwdef struct AgeSolution
    # Value Function
    ## End
    # Unshared V_end::Array{FLOAT, 4} = zeros(FLOAT, STATE_IDXs)
    ## Preshock
    V_end_perm::Array{FLOAT, 4} = zeros(FLOAT, N_Z, N_K, N_H, N_LOC)
    V_preshock_perm::Array{FLOAT, 4} = zeros(FLOAT, N_Z, N_K, N_H, N_LOC)
    V_preshock::Array{FLOAT, 4} = zeros(FLOAT, STATE_IDXs)
    ## Consume
    V_consume::Array{FLOAT, 4} = zeros(FLOAT, STATE_IDXs)
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
    λ_move_postmove::Array{FLOAT, 4} = zeros(FLOAT, N_K, N_Z, 1, N_LOC)
    ## Consumption
    λ_postc::Array{FLOAT, 4} = zeros(FLOAT, STATE_IDXs)
    ## Shock
    λ_preshock_perm::Array{FLOAT, 4} = zeros(FLOAT, N_Z, N_K, N_H, N_LOC)
end


"""
Data for a single age group slice, intended to be separately allocated for each
age group, so that this *can* be relied upon to represent an age group, even
after other age groups have been solved.
"""
@kwdef struct PeriodSolutionSlice
    # Population Distribution
    ## Price
    # Unshared λ_start
    λ_postprice::Array{FLOAT, 4} = zeros(FLOAT, STATE_IDXs)
    ## Sell
    λ_postsell::Array{FLOAT, 4} = zeros(FLOAT, STATE_IDXs)
    ## Move
    λ_postmove::Array{FLOAT, 4} = zeros(FLOAT, STATE_IDXs)
    ## Buy
    λ_postbuy::Array{FLOAT, 4} = zeros(FLOAT, STATE_IDXs)
    ## Income
    # Unshared λ_postincome
    ## Consumption
    ## Shock
    λ_postshock_mat::Array{FLOAT, 2} = zeros(FLOAT, N_Z, N_K*N_H*N_LOC)
    # Unshared λ_end
    ## Miscellaneous
    kloc_scratch::Vector{Array{FLOAT, 1}} = [zeros(FLOAT, N_K) for _ in 1:N_LOC]
    origin_weights::Array{FLOAT, 4} = zeros(FLOAT, N_K, N_Z, 1, N_LOC)
    P_sell::Array{FLOAT, 4} = zeros(FLOAT, STATE_IDXs)
    P_buy::Array{FLOAT, 4} = zeros(FLOAT, STATE_IDXs)
    #^^^ Everything above used to be in AgeSolution
    # Value Function
    ## Next
    V_end::Array{FLOAT, 4} = zeros(FLOAT, STATE_IDXs)
    ## Consume
    wealthi_postc_k_prec::Array{Int,4} = zeros(Int, STATE_IDXs)
    ## Income
    V_income::Array{FLOAT, 4} = zeros(FLOAT, STATE_IDXs)
    wealth_postinc_k_preinc::Array{FLOAT, 4} = zeros(FLOAT, STATE_IDXs)
    ## Buy
    V_choosebuy::Array{FLOAT, 4} = zeros(FLOAT, STATE_IDXs)
    ## Move
    eψV_move_tilde::Array{FLOAT, 4} = zeros(FLOAT, N_K, N_Z, 1, N_LOC)
    eψV_postmove_tilde::Array{FLOAT, 4} = zeros(FLOAT, N_K, N_Z, 1, N_LOC)
    ψV_means::Array{FLOAT, 4} = zeros(FLOAT, N_K, N_Z, 1, 1)
    V_move::Array{FLOAT, 4} = zeros(FLOAT, STATE_IDXs)
    ## Sell
    V_sell::Array{FLOAT, 4} = zeros(FLOAT, STATE_IDXs)
    V_choosesell::Array{FLOAT, 4} = zeros(FLOAT, STATE_IDXs)
    ## Start
    V_start::Array{FLOAT, 4} = zeros(FLOAT, STATE_IDXs)
    
    # Population Distribution
    λ_start::Array{FLOAT, 4} = zeros(FLOAT, STATE_IDXs)
    λ_postincome::Array{FLOAT, 4} = zeros(FLOAT, STATE_IDXs)
    λ_end::Array{FLOAT, 4} = zeros(FLOAT, STATE_IDXs)
end

"Precomputed data, for a given set of parameters."
@kwdef struct Precomputed
    wealth_postprice_k_preprice::Array{FLOAT, 4} = zeros(FLOAT, N_K, 1, N_H, N_LOC)
    wealth_postsell_k_presell::Array{FLOAT, 4} = zeros(FLOAT, N_K, 1, N_H, N_LOC)
    
    κqh_own::Array{FLOAT, 4} = zeros(FLOAT, 1, 1, N_H, N_LOC)
    forbidden_states::Array{FLOAT, 4} = zeros(FLOAT, N_K, 1, N_H, N_LOC)
    
    eψFu_inv::Array{FLOAT,2} = zeros(FLOAT, N_LOC, N_LOC)
    u_indirect_loc::Vector{FLOAT} = fill(NaN, N_LOC)
end


#############################
# Age-Period Data Container #
#############################
"""
A struct containing all the data that a particular age group slice needs to make their decisions.
"""
@kwdef struct AgeData{S}
    agei::Int
    age_solution::AgeSolution = AgeSolution()
    period_solution_slice::PeriodSolutionSlice = PeriodSolutionSlice()
    precomputed::Precomputed = Precomputed()
    prices::S = Prices()
end
Base.Broadcast.broadcastable(ad::AgeData) = Ref(ad)


#########################
# Period Data Container #
#########################
struct PeriodSolution
    period_solution_slices::Vector{PeriodSolutionSlice}
end
PeriodSolution() = PeriodSolution([PeriodSolutionSlice() for _ in 1:N_AGE])
Base.getindex(ps::PeriodSolution, i::Int) = ps.period_solution_slices[i]

@kwdef struct PeriodData{S}
    age_solution::AgeSolution = AgeSolution()
    period_solution::PeriodSolution = PeriodSolution()
    precomputed::Precomputed = Precomputed()
    prices::S = Prices()
end
Base.Broadcast.broadcastable(pd::PeriodData) = Ref(pd)

"""
Return views into the slice at agei of each field of PeriodData, as an AgeData object.
Slice and combine the preallocated data to get all necessary data for a single age slice.
"""
function AgeData(period_data::PeriodData, agei::Int)
    (;age_solution, period_solution, precomputed, prices) = period_data
    return AgeData(agei, age_solution, period_solution[agei], precomputed, prices)
end
Base.getindex(pd::PeriodData, agei::Int) = AgeData(pd, agei)
Base.lastindex(pd::PeriodData) = length(pd.period_solution.period_solution_slices)

######################
# Subfield Accessors #
######################

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
        return getproperty(getfield(pd, :prices), s)
    end
end

function Base.getproperty(params::Params, s::Symbol)
    if hasfield(Params, s)
        return getfield(params, s)
    else
        return getproperty(params.spatial, s)
    end
end
