
include("plot.jl")

"""
Generate reports and descriptive statistics from simulated data.
"""

################
# Helper Stuff #
################

idempotent_println(x) = (println(x); x)

function get_changes(path_data, s)
    s_mat = squeeze(getproperty(path_data, s))
    return s_mat[:,end] .- s_mat[:,1]
end
"This function is type-unstable, use only for debugging."
squeeze(A::AbstractArray) = reshape(A, filter(!=(1), size(A)))

#########################
# Exploring Simulations #
#########################

"Compare populations between two simulations."
POP_GRIDS_NAMES = [:λ_start,  :λ_move, :λ_nomove, :λ_move_k_postmove, :λ_postmove, :λ_premarket, :λ_postmarket, :λ_preshock, :λ_next]
function compare_pops(ad::AgeData, ad2::AgeData)
    for s in POP_GRIDS_NAMES
        maxdiff, idx = findmax(abs.(getproperty(ad, s) .- getproperty(ad2, s)))
        println("Largest difference in $s: $maxdiff at $idx")
    end
end

function report_rental_market(period_data::PeriodData, params::Params; loci=1)
    (;r_m) = params

    age_data = AgeData(period_data, 1)

    # Lowest-wealth household which chooses to own their home
    ki = findfirst(>(1), age_data.h_live_star[:,3,1,1,1])
    WEALTH_GRID_FLAT[ki]
    # Quite nice that people who already own are more willing to invest, since they're more
    # locked in to the location.

    inc_ki = get_income(WEALTH_GRID_FLAT[ki], Z_GRID_FLAT[3], H_LIVE_GRID_FLAT[1], H_LET_GRID_FLAT[7], loc_grid[1], params)
    int_ki = get_interest(WEALTH_GRID_FLAT[ki], H_LIVE_GRID_FLAT[1], H_LET_GRID_FLAT[7], loc_grid[1], params)

    # Even with people owning as much rental real estate as they can, returns are still way over the risk free rate.
    # Pretty interesting, not necessarily a problem except inasmuch as unrealistically poor people are owning
    # rental real estate. I was imagining it would only be worthwhile if you don't have to borrow to do so.
    # But maybe not. I'll just target the appropriate moments and see what I get.
    # If it's still realistic, I'll introduce an absentee landlord with a higher maintenance rate.
    # ^I should probably do that anyway.

    ρ = loc_grid[loci].ρ
    q = loc_grid[loci].q
    smallest_home = H_LET_GRID_FLAT[2]
    
    # Rent collected per period
    dec_rent = ρ*smallest_home*(1-params.landlord_tax)
    # Mortgage interest paid per period (at full mortgage)
    dec_interest_mortgage = q*smallest_home*r_m
    # Savings interest foregone per period (at zero mortgage)
    dec_interest_cash = q*smallest_home*r

    # Plot returns by wealth/h_let
    # Visualize joint h_live, h_let decisions
    # Add owner-investor to explore
end

function compare_rent_demand(pd::PeriodData, params; ρ1=36, ρ2=37)
    pd_ρ1 = deepcopy(pd)
    pd_ρ2 = pd

    pd_ρ1.loc_grid.ρ[1] = ρ1
    solve_household_problem_steady_state!(pd_ρ1, params)

    pd_ρ2.loc_grid.ρ[1] = ρ2
    solve_household_problem_steady_state!(pd_ρ2, params)

    get_rental_market_clearing(pd_ρ1, params)[2][1]
    get_rental_market_clearing(pd_ρ2, params)[2][1]

    (;loc_grid) = pd

    ad_ρ1 = AgeData(pd_ρ1, 6)
    ad_ρ2 = AgeData(pd_ρ2, 6)

    h_rent_star_ρ1 = get_h_rent_star(pd_ρ1, params)
    h_rent_star_ρ2 = get_h_rent_star(pd_ρ2, params)
    λ_ρ1 = stack(pd_ρ1.λ_prec)
    λ_ρ2 = stack(pd_ρ2.λ_prec)
    H_rent_ρ1 = h_rent_star_ρ1 .* λ_ρ1
    H_rent_ρ2 = h_rent_star_ρ2 .* λ_ρ2

    h_rent_star_diff = h_rent_star_ρ2 .- h_rent_star_ρ1
    λ_diff = λ_ρ2 .- λ_ρ1
    H_rent_diff = H_rent_ρ2 .- H_rent_ρ1

    #_, idx = findmax(H_rent_diff)
    return h_rent_star_diff, λ_diff, H_rent_diff
end

###################
# Data Statistics #
###################

get_mean_wealth_path(pd::PeriodData) = get_mean_wealth.(AgeData.(Ref(pd), 1:N_AGE))
