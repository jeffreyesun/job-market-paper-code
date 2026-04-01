

##############
# Precompute #
##############

const ONES_MAT = ones(N_LOC, N_LOC)
const ONES_MAT_GPU = CUDA.ones(N_LOC, N_LOC)
get_ones_mat(::AbstractArray) = ONES_MAT
get_ones_mat(::CuArray) = ONES_MAT_GPU

"Precompute data for a given set of prices and parameters."
function precompute!(precomp, prices::Union{Prices, PricesGPU}, params::Params)
    (;γ, σ, ψ, F_u_fixed, F_u_dist, ϕ) = params
    (;α) = params.spatial
    (;ρ, q, q_last) = prices
    (;
        wealth_postprice_k_preprice, wealth_postsell_k_presell,
        κqh_own, forbidden_states,
        eψFu_inv, u_indirect_loc,
    ) = precomp

    # Compute origin-destination moving costs (fixed cost does not apply to moving within a location)
    dist_loc = get_dist_loc(q)
    ones_mat = get_ones_mat(q)
    ψFu = ψ.*F_u_fixed.*(ones_mat-I)
    ψFu .+= F_u_dist.*dist_loc
    @. eψFu_inv = 1/exp(ψFu)
    # eFu_inv represents the attractiveness of a destination relative to an origin (higher values are better)
    # [origin, destination]

    # Compute forbidden states
    # Note: this κ is 1-κ in the paper
    h_grid = get_h_grid(q)
    wealth_grid = get_wealth_grid(q)
    @. κqh_own = params.κ * prices.q * h_grid
    @. forbidden_states = -Inf*(wealth_grid < κqh_own)

    # Precompute indirect utility
    u_indirect_loc .= vec(get_indirect_u.(1, ρ, α, γ, σ))

    # Precompute wealth effects of price change and selling
    @. wealth_postprice_k_preprice = get_wealth_postprice(wealth_grid, h_grid, q, q_last)
    @. wealth_postsell_k_presell = get_wealth_postsell(wealth_grid, h_grid, q, ϕ)

    return precomp
end

"Precompute"
function precompute!(pd::Union{PeriodData,PeriodDataGPU}, params::Params)
    (;precomputed, prices) = pd
    precompute!(precomputed, prices, params)
    return pd
end


###############
# Set Up Data #
###############

function set_climate!(params::Params; decade=nothing, scenario_ind=nothing)
    (;A_bar, A_g, α_bar, α_g, δ_bar, δ_g) = params
    scenario_ind = something(scenario_ind, params.scenario_ind)
    decade = something(decade, params.decade)

    SST = scenario_ind == -1 ? 0 : SST_GRID[scenario_ind, decade]

    @. params.A = A_bar * exp(A_g*SST)
    @. params.α = α_bar * exp(α_g*SST)
    @. params.δ = δ_bar + δ_g*SST
    return params
end

"Set prices and params on PeriodData"
function set_prices_and_params!(prealloc::Union{PeriodData,AgeData,PeriodDataGPU,AgeDataGPU}, params; q=nothing, ρ=nothing, q_last=nothing, H_S=nothing, H_S_prior=nothing, decade=nothing, scenario_ind=nothing)
    
    # Prices #
    if !isnothing(q)
        isnothing(q_last) && error("You must specify q_last when setting q")

        prealloc.q[:] .= q[:]
        prealloc.q_last[:] .= q_last[:]
    end
    if !isnothing(ρ)
        prealloc.prices.ρ[:] .= ρ[:]
    end
    (;q, q_last, ρ) = prealloc.prices

    # Housing Supply #
    if isnothing(H_S)
        if isnothing(H_S_prior)
            error("To compute H_S as in steady state, you must explicitly set H_S_prior=0")
        else
            get_H_S!(params, q, H_S_prior)
        end
    else
        params.spatial.H_S .= H_S
    end

    # Climate #
    set_climate!(params; decade, scenario_ind)

    # Precomputation #
    precompute!(prealloc.precomputed, prealloc.prices, params)

    return prealloc
end

"Preallocate and precompute"
function PeriodData(prices, params::Params; q=prices.q, ρ=prices.ρ, q_last=prices.q_last, H_S=nothing, H_S_prior=0)
    return pd = set_prices_and_params!(PeriodData(;prices), params; q, ρ, q_last, H_S, H_S_prior)
end

PeriodData(params::Params; kwargs...) = PeriodData(Prices(), params; kwargs...)
