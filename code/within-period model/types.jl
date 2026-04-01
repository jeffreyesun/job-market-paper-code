
# Location #
#---------#

@kwdef struct LocalPrices
    q::FLOAT=NaN
    ρ::FLOAT=NaN
    q_last::FLOAT=NaN
end
LocalPrices(q, ρ) = LocalPrices(;q, ρ)

@kwdef struct Location
    GISMATCH::Int64 = 0

    # Current Conditions #
    A::FLOAT=NaN
    α::FLOAT=NaN
    δ::FLOAT=NaN # '000 2020 USD/decade/bedroom-equivalent
    H_S::FLOAT=NaN

    # Fundamentals #
    # Housing
    elasticity::FLOAT=NaN # Housing supply elasticity
    Π::FLOAT=NaN # Housing supply shifter

    # Climate
    ## Intercept
    A_bar::FLOAT=NaN
    α_bar::FLOAT=NaN
    δ_bar::FLOAT=NaN
    ## Slope (per °C global warming)
    A_g::FLOAT=NaN
    α_g::FLOAT=NaN
    δ_g::FLOAT=NaN
end
Location(row::DataFrameRow) = Location(; row...)


@kwdef struct LocationMoments
    GISMATCH::Int64
    pop::FLOAT
    H::FLOAT
    housing_price::FLOAT
    homeown_frac::FLOAT
    total_ownocc_valueh::FLOAT
    mean_rent::FLOAT
    mean_earn::FLOAT
    mean_earn_workage::FLOAT
    #median_rent_to_earn::FLOAT
    mean_income::FLOAT
    mean_income_workage::FLOAT
end

# Aggregate Moments. All should be in unitless. Flows are decadal.
@kwdef struct AggMoments
    rent_share_earn::FLOAT                # γ, weight of housing in consumption
    homeown_rate::FLOAT                   # χ_let, maintenance cost on rental property
    share_H_sold::FLOAT                   # ϕ, realtor fees
    share_pop_moving::FLOAT               # F_u_fixed, fixed utility cost of moving
    share_young_moving::FLOAT             # F_m, monetary cost of moving
    mean_wealth_homeown_over_mean::FLOAT  # r_m, mortgage interest rate
    mean_wealth_oldest_over_mean::FLOAT   # bequest_motive, bequest motive strength
end

# Parameters #
#------------#

@kwdef mutable struct Params{S} # All flows are decadal
    γ::FLOAT = FLOAT(0.04968)           # Weight of h in consumption
    σ::FLOAT = FLOAT(0.7)               # Elasticity of substitution between housing and goods
    #η::FLOAT = FLOAT(1.1)              # Risk aversion
    ϕ::FLOAT = FLOAT(0.07)              # Realtor fees
    κ::FLOAT = FLOAT(0.2)               # Equity Requirement
    ξ::FLOAT = FLOAT(2.0)               # Housing choice elasticity (inverse taste shock variance)
    F_u_fixed::FLOAT = FLOAT(0.5588)    # Utility moving cost, fixed part
    F_u_dist::FLOAT = FLOAT(0.004)      # Utility moving cost, distance part
    χ::FLOAT = FLOAT(0.3622)            # Maintenance cost, rental property ('000 2020 USD/decade/bedroom-equivalent)
    r_m::FLOAT = FLOAT(0.8)             # Mortgage interest rate (decadal)
    bequest_motive::FLOAT = FLOAT(1.71) # Bequest motive strength
    ψ::FLOAT = FLOAT(4.0)               # Migration elasticity (inverse preference shock variance)
    δ_dep::FLOAT = FLOAT(0.05)          # Depreciation rate of location-level housing stock
    decade::Int = 1                     # Decade of the period in question
    scenario_ind::Int = -1              # Index of climate scenario (-1 means pre-climate change baseline)
    spatial::S                          # Location-level parameters
end
Params(s) = Params(;spatial=s)
Base.Broadcast.broadcastable(params::Params) = Ref(params)

###################
# Type Shorthands #
###################

StructArray{LocalPrices, 4}(n_loc::Int=N_LOC) = StructArray(map(x->LocalPrices(), ones(1,1,1,n_loc)))
Prices = StructArray{LocalPrices, 4}

StructArray{Location, 4}(n_loc::Int=N_LOC) = StructArray(map(x->Location(), ones(1,1,1,n_loc)))
LocGrid = StructArray{Location, 4}
