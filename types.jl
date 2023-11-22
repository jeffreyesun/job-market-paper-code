
FLOAT_PRECISION = Float32

# Location #
#---------#

@kwdef struct Location{T}
    GISMATCH::Int64 = 0

    # Current Conditions #
    # Exogenous
    A::T=NaN
    α::T=NaN
    δ::T=NaN

    # Equilibrium
    ## Prices
    q::T=NaN
    ρ::T=NaN
    ## Quantities
    pop::T=NaN
    H_D::T=NaN # Housing quantity *demanded* (not necessarily supplied)
    ## History
    q_last::T=NaN

    # Fundamentals #
    # Housing
    H::T=NaN # Housing supply
    Π::T=NaN # Housing supply shifter (Π⁻¹ in Greaney)
    H_bar::T=NaN # Housing supply intercept
    elasticity::T=NaN # Housing supply elasticity

    # Climate
    ## Intercept
    A_bar::T=NaN
    α_bar::T=NaN
    δ_bar::T=NaN
    ## Slope (per °C global warming)
    A_g::T=NaN
    α_g::T=NaN
    δ_g::T=NaN
end
Location(args...) = Location{FLOAT_PRECISION}(args...)
Location(;kwargs...) = Location{FLOAT_PRECISION}(;kwargs...)
Location(GISMATCH, A, α, δ, q, ρ, pop, H, q_last) = Location(;GISMATCH, A, α, δ, q, ρ, pop, H, q_last)
#Location{T}(GISMATCH) where T = Location{T}(;GISMATCH)

@kwdef struct LocationMoments{T}
    GISMATCH::Int64
    homeown_frac::T
    total_ownocc_valueh::T
    mean_rent::T
    mean_earn::T
    mean_earn_workage::T
    median_rent_to_earn::T
end
LocationMoments(args...; kwargs...) = LocationMoments{FLOAT_PRECISION}(args...; kwargs...)
LocationMoments(; kwargs...) = LocationMoments{FLOAT_PRECISION}(; kwargs...)

# Aggregate Moments. All should be in unitless. Flows are decadal.
@kwdef struct AggMoments{T}
    rent_share_earn::T                # γ, weight of housing in consumption
    homeown_rate::T                   # χ_let, maintenance cost on rental property
    share_H_sold::T                   # ϕ, realtor fees
    share_pop_moving::T               # F_u_fixed, fixed utility cost of moving
    share_young_moving::T             # F_m, monetary cost of moving
    mean_wealth_homeown_over_mean::T  # r_m, mortgage interest rate
    mean_wealth_oldest_over_mean::T   # bequest_motive, bequest motive strength
end

# Parameters #
#------------#

@kwdef struct Params{T} # All flows are decadal
    γ::T = 0.2              # Weight of h in consumption
    σ::T = 2.0              # Elasticity of substitution between housing and goods
    η::T = 2.0              # Risk aversion
    ϕ::T = 0.07             # Realtor fees
    κ::T = 0.2              # Equity Requirement
    F_u_fixed::T = 5.1      # Utility moving cost, fixed part
    F_u_dist::T = 1.1e-3    # Utility moving cost, distance part
    F_m::T = 13.3           # Monetary moving cost
    χ_live::T = 10          # Maintenance cost, owner-occupied ('000 2020 USD/decade/bedroom-equivalent)
    χ_let::T = 20           # Maintenance cost, rental property ('000 2020 USD/decade/bedroom-equivalent)
    r_m::T = 0.8            # Mortgage rate (decadal)
    bequest_motive::T = 2.6 # Bequest motive strength
    ψ::T = 5.0              # Migration elasticity
    δ_dep::T = 0.05         # Depreciation rate of housing
end
Params(args...; kwargs...) = Params{FLOAT_PRECISION}(args...; kwargs...)
Params(;kwargs...) = Params{FLOAT_PRECISION}(;kwargs...)
Base.Broadcast.broadcastable(params::Params) = Ref(params)

# Climate #
#---------#
"""
logw1 = logw + g_w
logm1 = logm + β_m*logw + ε
emissions = w/m

SST1 = ρ_SST * SST + β_SST * emissions
δ1   = ρ_δ   * δ   + β_δ   * emissions

Rule (to notes): SST and δ represent deviations from pre-industrial baseline throughout
"""
@kwdef struct ClimateParams{T}
    # Initial Values
    w0::T     = 106_715*10.0 # Initial energy generation (TWh/decade)
    m0::T     = 0.35478    # Initial carbon efficiency (kWh/kg) (equiv. MWh/t or TWh/Mt)
    SST0::T   = 0.45       # Initial SST (°C, deviation from pre-industrial baseline)
    δ_agg0::T = 0.0        # Initial δ_agg (2020 USD per unit of housing, deviation from pre-industrial baseline)

    # Evolution
    gw::T      = 0.0155*10 # Log trend in energy generation (decadal)
    β_m::T     = 0.26     # m sensitivity to w
    ρ_SST::T   = 0.9       # SST persistence
    β_SST::T   = 5e-4      # SST sensitivity to emissions (°C/Gt)
    #ρ_δ_agg::T = 0.9       # δ_agg persistence
    #β_δ_agg::T = 5e-4      # δ_agg sensitivity to emissions (2020 USD/Gt)

    # Shocks
    σ_m::T = 0.6           # Standard deviation of carbon efficiency shock
end
ClimateParams(;kwargs...) = ClimateParams{FLOAT_PRECISION}(; kwargs...)

@kwdef struct ClimateState{T}
    logw::T = NaN       # Log of energy generation (TWh/decade)
    logm::T = NaN       # Log of carbon efficiency (kg/kWh)
    emissions::T = NaN  # Emissions (Gt/decade)
    SST::T = NaN        # SST (°C, deviation from pre-industrial baseline)
    ΔSST::T = NaN       # SST (°C, deviation from baseline year)
    ε_m::T = NaN        # Carbon efficiency shock
    #δ_agg::T = NaN      # δ_agg (2020 USD per unit of housing, deviation from pre-industrial baseline)
end
ClimateState(;kwargs...) = ClimateState{FLOAT_PRECISION}(; kwargs...)

# Scheduler #
#-----------#

@kwdef mutable struct JeffreysReallyBadScheduler{T}
    # Parameters
    ε::T = 2e-3
    "Tolerance for error decrease."
    g_tol::T = 1e-4
    stepsize_change_factor::T = 2.0
    
    # Conditions on stepsize updates
    "Minimum proportion of error that can decrease each iteration without triggering stepsize increase."
    convergence_speed_threshold::T = 0.01
    max_iterations_without_global_progress = 10
    max_iterations_without_local_progress = 10
    max_iterations_with_slow_progress = 6
    
    # Public state
    stepsize::T = 1e-2
    stop::Bool = false
    #n_iter::Int = 0

    # Error Tracking
    error::T = NaN
    error_last::T = NaN
    error_min::T = NaN
    n_iterations::Int = 0
    n_stepsize_decreased::Int = 0
    n_stepsize_increased::Int = 0

    # Short-Term Progress Tracking
    iterations_without_global_progress::Int = 0
    iterations_without_local_progress::Int = 0
    iterations_with_global_progress::Int = 0
    iterations_with_local_progress::Int = 0
    iterations_with_slow_progress::Int = 0
    iterations_oscillating::Int = 0
    iterations_without_consistent_progress::Int = 0
end
JeffreysReallyBadScheduler(; kwargs...) = JeffreysReallyBadScheduler{FLOAT_PRECISION}(; kwargs...)


###################
# Type Shorthands #
###################

LocGrid{T} = StructArray{Location{T}, 5}
"""
Note: LocGrid{T, 5} is not a concrete type. For best performance, make sure to pass loc_grid
into a function as a separate input. If you unpack it on-the-fly from, e.g., AgeData, then
type-inference will fail.
I should just make LocGrid constant or use it as a type parameter for AgeData, but it's
my project and I want all my type signatures nice and simple.
"""
LocGrid{T}(n_loc=N_LOC) where T = StructArray(map(x->Location{T}(), ones(1,1,1,1,n_loc)))
LocGrid(args...) = LocGrid{FLOAT_PRECISION}(args...)

LocGridPath{T} = StructArray{Location{T}, 8}
ClimatePath{T} = StructArray{ClimateState{T}, 8}