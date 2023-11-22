
##################
# Initialization #
##################

# Location Grid #
#---------------#

"Get the initial guesses of calibrated values for each location: rent ρ, productivity A, and α."
function get_calibration_loc_grid_init(loc_grid_empirical::LocGrid, loc_target_moments, params::Params)
    (;γ, χ_live, χ_let) = params
    loc_grid = deepcopy(loc_grid_empirical)

    @. loc_grid.ρ .= loc_grid.q * r + loc_grid.δ + χ_let - χ_live
    loc_grid.A .= normalize(loc_target_moments.mean_earn_workage, 1).*N_LOC
    @. loc_grid.α = loc_grid.ρ^γ / loc_grid.A
    loc_grid.α ./= mean(loc_grid.α)
    @assert all(>(0), loc_grid.α)

    loc_grid.H .= NaN
    return loc_grid
end


#####################
# Simulated Moments #
#####################

function get_h_rent_star(period_data::PeriodData, params::Params)
    (;wealthi_postc_k_prec, loc_grid) = period_data
    (;ρ) = loc_grid
    (;γ, σ) = params

    # Demand for rental housing (supply of rental real estate assets)
    h_by_ki = get_h_demand.(WEALTH_GRID .- WEALTH_NEXT_GRID, ρ, γ, σ)
    
    h_rent_star = zeros(FLOAT_PRECISION, STATE_IDXs_FULL)
    for agei=1:N_AGE
        for idx=CartesianIndices((N_K,N_Z,1,N_Hle,N_LOC))
            ki = idx[1]
            loci = idx[5]
            ki_postc = wealthi_postc_k_prec[agei][idx]
            h_rent_star[idx,agei] = h_by_ki[ki,1,1,1,loci,1,ki_postc]
        end
    end
    return h_rent_star
end

function get_rent_share_of_income_loc(h_rent_to_income, λ_prec)
    r_i_order = sortperm(vec(h_rent_to_income))
    λ_quantiles = cumsum(vec(λ_prec)[r_i_order])
    λ_quantiles ./= λ_quantiles[end]
    median_idx = findfirst(>(0.5), λ_quantiles)
    left_weight = λ_quantiles[median_idx] - 0.5f0
    right_weight = 0.5f0 - λ_quantiles[max(median_idx-1, 1)]
    median_r2i_right = h_rent_to_income[r_i_order[median_idx]]
    median_r2i_left = h_rent_to_income[r_i_order[max(median_idx-1, 1)]]
    return (left_weight * median_r2i_left + right_weight * median_r2i_right) / (left_weight + right_weight)
end

function get_rent_share_of_income(period_data::PeriodData, params::Params)
    (;λ_prec) = period_data
    (;ρ) = period_data.loc_grid
    h_rent_star = get_h_rent_star(period_data, params)
    h_rent_to_income = h_rent_star .* ρ ./ stack(Z_GRID)
    λ_prec_full = stack(λ_prec)
    median_rent_to_income = [
        get_rent_share_of_income_loc(h_rent_to_income[:,:,1,:,loci,:], λ_prec_full[:,:,1,:,loci,:]) for loci=1:N_LOC
    ]
    return pad_dims(median_rent_to_income; left=LOC_DIM-1)
end

function get_homeownership_rate(period_data::PeriodData)
    (;λ_prec) = period_data
    renter_mass = sum([sum(λ_prec_agei[:,:,1,:,:]) for λ_prec_agei=λ_prec])
    return 1 - renter_mass / sum(sum.(λ_prec))
end

"Compute the share of housing sold each period for a certain age group."
function get_H_sold(ad::AgeData)
    (;λ_start, sell_live, sell_let) = ad

    λ_postprice = get_λ_postprice(λ_start, ad)
    _, λ_nomove = get_λ_move_nomove(λ_postprice, ad)

    λ_h_live_presell = λ_nomove .* H_LIVE_GRID
    H_live_sold_living = sum(λ_h_live_presell .* sell_live)
    H_live_presell_total = sum(λ_h_live_presell)

    λ_h_let_presell = λ_nomove .* H_LET_GRID
    H_let_sold_living = sum(λ_h_let_presell .* sell_let)
    H_let_presell_total = sum(λ_h_let_presell)

    return (;H_live_sold_living, H_live_presell_total, H_let_sold_living, H_let_presell_total)
end

"Compute the share of houses sold each period."
function get_share_H_sold(period_data::PeriodData)
    (;λ_prec) = period_data
    
    # Because these reuse the preallocated age_solution, this must not be parallelized
    H_sold_agevec = [get_H_sold(AgeData(period_data, agei)) for agei=1:N_AGE]
    # H sold by living households
    H_sold_living = Dict(k => sum(getproperty.(H_sold_agevec, k)) for k in keys(first(H_sold_agevec))) |> NamedTuple
    (;H_live_sold_living, H_let_sold_living, H_live_presell_total, H_let_presell_total) = H_sold_living

    # H sold by dying households
    H_sold_dying = sum(λ_prec[N_AGE] .* (H_LIVE_GRID .+ H_LET_GRID))

    # Total H sold and total H
    H_sold_total = H_live_sold_living + H_let_sold_living + H_sold_dying
    H_total = H_live_presell_total + H_let_presell_total + H_sold_dying

    return share_H_sold = H_sold_total / H_total
end

get_share_moving(period_data::PeriodData) = only(get_mean(period_data, :P_move, :λ_start, (1,2,3,4,5)))

function get_share_young_moving(pd::PeriodData)
    mean([only(get_weightedmean(AgeData(pd, agei), :P_move, :λ_start, (1,2,3,4,5))) for agei=(1,2)])
end

function get_mean_wealth(pd::PeriodData)
    (;λ_start) = pd
    return get_weightedsum(WEALTH_GRID, λ_start, (1,2,3,4,5)) ./ sum_perioddata(λ_start, (1,2,3,4,5)) |> only
end

get_mean_wealth(ad::AgeData) = sum(WEALTH_GRID .* ad.λ_start) / sum(ad.λ_start)

function get_mean_wealth_by_homeownership(pd::PeriodData)
    (;λ_start) = pd
    wealth_totals = get_weightedsum(WEALTH_GRID, λ_start, (1,2,4,5))
    pop_totals = sum_perioddata(λ_start, (1,2,4,5))

    meanwealth_own = sum(wealth_totals[:,:,2:end,:,:]) ./ sum(pop_totals[:,:,2:end,:,:])
    meanwealth_rent = sum(wealth_totals[:,:,1,:,:]) ./ sum(pop_totals[:,:,1,:,:])
    return meanwealth_own / meanwealth_rent
end

function get_mean_wealth_oldest(pd::PeriodData)
    return get_mean_wealth(AgeData(pd, N_AGE)) / get_mean_wealth(pd)
end

function get_simulated_moments(period_data, params::Params)
    return AggMoments(
        rent_share_earn = mean(get_rent_share_of_income(period_data, params)),
        homeown_rate = get_homeownership_rate(period_data),
        #share_H_sold = get_share_H_sold(period_data),
        share_H_sold = 0.6513f0,
        share_pop_moving = get_share_moving(period_data),
        share_young_moving = get_share_young_moving(period_data),
        mean_wealth_homeown_over_mean = get_mean_wealth_by_homeownership(period_data),
        mean_wealth_oldest_over_mean = get_mean_wealth_oldest(period_data),
    )
end

##############################
# Market Clearing Conditions #
##############################

function get_rental_market_clearing(period_data::PeriodData, params::Params)
    (;λ_prec, wealthi_postc_k_prec, loc_grid) = period_data
    (;ρ) = loc_grid
    (;γ, σ) = params

    # Supply of rental housing (demand for rental real estate assets)
    H_let = get_H_let(period_data)

    # Demand for rental housing (supply of rental real estate assets)
    h_by_ki = get_h_demand.(WEALTH_GRID .- WEALTH_NEXT_GRID, ρ, γ, σ)
    h_by_ki_small = reshape(h_by_ki, N_K, N_LOC, N_K)
    
    H_rent = zeros(FLOAT_PRECISION, (1,1,1,1,N_LOC))
    for loci=1:N_LOC
        for agei=1:N_AGE
            for idx=CartesianIndices((N_K,N_Z,1,N_Hle,loci:loci))
                ki = idx[1]
                ki_postc = wealthi_postc_k_prec[agei][idx]
                h = h_by_ki_small[ki,loci,ki_postc]
                λ = λ_prec[agei][idx]
                H_rent[loci] += h*λ
            end
        end
    end

    return H_let, H_rent
end
function get_excess_H_rent(period_data::PeriodData, params::Params)
    H_let, H_rent = get_rental_market_clearing(period_data, params)
    return H_rent .- H_let
end

function get_real_estate_market_clearing(period_data::PeriodData, params::Params)
    (;loc_grid) = period_data

    H_let  = get_H_let(period_data)  # Demand for rental real estate assets
    H_live = get_H_live(period_data) # Demand for owner-occupied real estate assets
    
    H_D = H_live .+ H_let # Overall demand for real estate assets
    #H_S = get_H_construction.(loc_grid) # Steady-state supply of real estate assets
    H_S = loc_grid.H
    return H_D, H_S
end
function get_excess_H_S(period_data::PeriodData, params::Params)
    H_D, H_S = get_real_estate_market_clearing(period_data, params)
    return H_S .- H_D
end
get_excess_H_D(pd::PeriodData, params::Params) = -get_excess_H_S(pd, params)

function get_excess_demand(period_data::PeriodData, params::Params)
    excess_H_D = get_excess_H_D(period_data, params)
    excess_H_rent = get_excess_H_rent(period_data, params)
    return excess_H_D, excess_H_rent
end


##############################################
# Steady State Given Prices and Fundamentals #
##############################################

# Initial Distribution of Young Households #
#------------------------------------------#

function get_λ_init(young_loc)
    young_wealth = YOUNG_WEALTH_INIT
    young_z = reshape(YOUNG_Z_FLAT, 1, N_Z)
    young_h_li = reshape(e_vec(N_Hli, 1), 1,1,N_Hli)
    young_h_le = reshape(e_vec(N_Hle, 1), 1,1,1,N_Hle)

    return young_wealth .* young_z .* young_h_li .* young_h_le .* young_loc
end

"Get initial distribution of young households, conditional on location, in steady state."
function get_λ_init_steady_state(period_data::PeriodData, params::Params)
    (;ψV_means) = period_data
    (;ψ) = params
    V_start = period_data.V_price
    eψV_start = @. exp(ψ*V_start[1][:,:,1,1,:] - ψV_means[1])
    eψV_start .*= N_LOC ./ sum(eψV_start; dims=3)
    young_loc = reshape(eψV_start, N_K, N_Z, 1, 1, N_LOC)

    return get_λ_init(young_loc)
end

"""
Get initial distribution of young households, assuming the spatial distribution to be equal to
the contemporaneous spatial distribution of agei-4 (age 60-70) households.
"""
function get_λ_init_transition(period_data::PeriodData, params::Params)
    (;λ_start) = period_data
    young_loc = sum(λ_start[2], dims=(1,2,3,4))
    return get_λ_init(young_loc)
end

# Household's Problem Solution #
#------------------------------#
function update_params!(pd::PeriodData, param_updates)
    (;loc_grid) = pd
    for s in keys(param_updates)
        if hasfield(Location, s)
            getproperty(loc_grid, s) .= param_updates[s]
        elseif hasfield(Params, s)
            error("Updating parameters is not implemented yet.")
        else
            error("Unknown field $s.")
        end
    end
end

fill_in_q_last_steady_state!(pd::PeriodData) = (pd.q_last .= pd.q; pd)

"""
Solve the household's problem in steady-state, given prices. Yields the household's full
value function and state distribution. Market-clearing conditions are not enforced.
"""
function solve_household_problem_steady_state!(
        period_data::PeriodData, params::Params; param_updates...
    )
    update_params!(period_data, param_updates)
    fill_in_q_last_steady_state!(period_data)
    precompute!(period_data, params)

    solve_V_backward_steady_state!(period_data, params)
    return simulate_forward_steady_state!(period_data, params)
end

function solve_household_problem_steady_state(loc_grid::LocGrid, params::Params; kwargs...)
    return solve_household_problem_steady_state!(PeriodData(;loc_grid), params; kwargs...)
end

# Extract and Precompute Data #
#-----------------------------#

"Extract and solve a given age group, given an already-solved PeriodData."
function get_age_data_solved!(pd::PeriodData, agei::Int, params::Params)
    ad = AgeData(pd, agei)
    solve_period!(ad, ad.V_next, params)
    iterate_λ(ad.λ_start, ad, params)
    return ad
end


#######################################################
# Steady State Market Clearing and Moment Calibration #
#######################################################

# Partial Market Clearing #
#-------------------------#
"Solve for steady-state rent ρ, holding real estate prices q constant."
function solve_rent_steady_state!(period_data, params, ρ; stepsize=1e-2, ε=1e-4)
    scheduler = JeffreysReallyBadScheduler(; stepsize, ε)

    while !scheduler.stop
        # Apply guesses
        period_data = solve_household_problem_steady_state!(period_data, params; ρ)

        # Compute moments
        H_let, H_rent = get_rental_market_clearing(period_data, params)
        excess_H_rent = H_rent .- H_let

        # Compute errors
        error = mean(x->x^2, excess_H_rent)
        stepsize = next_stepsize!(scheduler, error, true)

        # Update guesses
        ρ .+= excess_H_rent.*stepsize
    end

    return period_data
end

"Solve for steady-state real estate prices q, holding rent ρ constant."
function solve_q_steady_state!(period_data::PeriodData, params::Params, q=period_data.q; stepsize=1e-2, ε=1e-4)
    scheduler = JeffreysReallyBadScheduler(; stepsize, ε)

    while !scheduler.stop
        # Apply guesses
        period_data = solve_household_problem_steady_state!(period_data, params; q)

        # Compute moments
        H_D, H_S = get_real_estate_market_clearing(period_data, params)
        excess_H = H_S .- H_D
        
        # Compute errors
        error = mean(x->x^2, excess_H)
        stepsize = next_stepsize!(scheduler, error, true)

        # Update guesses
        q .-= excess_H.*stepsize
    end
    return period_data
end

# Partial Moment Calibration #
#----------------------------#
"Solve for steady-state α by matching given empirical populations."
function calibrate_steady_state_α!(period_data, params, α, pop_loc)
    scheduler = JeffreysReallyBadScheduler(; stepsize=1e-2, ε=1e-4)

    while !scheduler.stop
        # Apply guesses
        period_data = solve_household_problem_steady_state!(period_data, params; α)

        # Compute moments
        pop_sim = sumpop(period_data)
        excess_pop = pop_sim .- pop_loc

        # Compute errors
        error = mean(x->x^2, excess_pop)
        stepsize = next_stepsize!(scheduler, error, true)

        # Update guesses
        α .-= excess_pop.*stepsize
    end

    return period_data
end

# Full Location Moment Calibration #
#----------------------------------#
"Solve for rent ρ and amenities α, given known equilibrium prices q and populations."
function calibrate_steady_state_rent_and_α!(
        period_data::PeriodData, params::Params, pop_loc_target;
        ρ=copy(period_data.loc_grid.ρ), α=copy(period_data.loc_grid.α),
        stepsize=1e-4, ε=2e-4
    )
    (;γ, σ) = params
    pop_loc_target = normalize(pop_loc_target, 1) .* N_LOC
    
    scheduler = JeffreysReallyBadScheduler(; stepsize, ε)
    
    while !scheduler.stop
        # Apply guesses
        period_data = solve_household_problem_steady_state!(period_data, params; ρ, α)
        validate_loc_grid(period_data)

        # Compute moments
        excess_H_rent = get_excess_H_rent(period_data, params)
        pop_sim = sumpop(period_data)
        excess_pop = pop_sim .- pop_loc_target

        # Compute errors
        error_rent = mean(x->x^2, excess_H_rent)
        error_pop = mean(x->x^2, excess_pop)
        error_total = error_rent + error_pop
        stepsize = next_stepsize!(scheduler, error_total, true)

        # Update guesses
        price_amenity_old = @. get_price_level(ρ/500, γ, σ)^(σ/(1-σ))
        @. ρ += excess_H_rent*stepsize * 10
        price_amenity_new = @. get_price_level(ρ/500, γ, σ)^(σ/(1-σ))
        
        @. α /= price_amenity_new / price_amenity_old # Cancel out ρ effect on population
        α ./= mean(α)
        @. α -= excess_pop*stepsize .* 0.1

        #display(explore(AgeData(period_data, 1)))
    end

    return period_data
end

# Non-Spatial Parameter Calibration #
#-----------------------------------#
get_moment_vector(am::AggMoments) = collect(getfield.(Ref(am), fieldnames(AggMoments)))

import Base: -
function -(am1::AggMoments, am2::AggMoments)
    AggMoments((get_moment_vector(am1) .- get_moment_vector(am2))...)
end

χ_LET_STEPSIZE = 1
function update_params(params::Params, sim_moments::AggMoments, target_moments::AggMoments, stepsize)
    moment_error = sim_moments - target_moments

    @reset params.γ -= 0.1 * stepsize * moment_error.rent_share_earn
    @reset params.χ_let -= χ_LET_STEPSIZE * stepsize * moment_error.homeown_rate
    @reset params.ϕ += 0.1 * stepsize * moment_error.share_H_sold
    @reset params.F_u_fixed += 0.1 * stepsize * moment_error.share_pop_moving
    @reset params.F_m += 0.1 * stepsize * moment_error.share_young_moving
    @reset params.F_m += 0.1 * stepsize * moment_error.share_pop_moving
    @reset params.r_m -= 0.1 * stepsize * moment_error.mean_wealth_homeown_over_mean
    @reset params.bequest_motive -= 1 * stepsize * moment_error.mean_wealth_oldest_over_mean

    global params_last = params

    return params
end

"Apply the equilibrium effect of χ_let on rent without re-solving everything."
function apply_price_change_effects!(loc_grid::LocGrid, stepsize, moment_error)
    loc_grid.ρ .-= χ_LET_STEPSIZE * stepsize * moment_error.homeown_rate
    return loc_grid
end

function solve_params_steady_state(period_data::PeriodData, params_init::Params, target_moments::AggMoments; stepsize=1e-4, ε=1e-4)
    (;loc_grid) = period_data
    error = Inf
    params = deepcopy(params_init)

    while error > ε
        # Apply guesses
        period_data = solve_household_problem_steady_state!(period_data, params)

        # Compute moments
        sim_moments = get_simulated_moments(period_data, params)

        # Compute errors
        moment_error = sim_moments - target_moments
        error = mean(abs.(get_moment_vector(moment_error)))

        # Update guesses
        params = update_params(params, sim_moments, target_moments, stepsize)
        apply_price_change_effects!(loc_grid, stepsize, moment_error)

        # Report progress
        @show error
    end

    return params
end

# Full Moment Calibration #
function calibrate!(
        period_data::PeriodData, params_init::Params, pop_loc_target, target_moments::AggMoments,
        ;ρ=period_data.loc_grid.ρ, α=period_data.loc_grid.α, stepsize=1e-4, ε=1e-4
    )
    (;loc_grid) = period_data
    params = deepcopy(params_init)
    pop_loc_target = normalize(pop_loc_target, 1) .* N_LOC
    scheduler = JeffreysReallyBadScheduler(; stepsize, ε)

    while !scheduler.stop
        # Apply guesses
        period_data = solve_household_problem_steady_state!(period_data, params; ρ, α)
        validate_loc_grid(period_data)
        
        # Compute moments
        sim_moments = get_simulated_moments(period_data, params)
        excess_H_rent = get_excess_H_rent(period_data, params)
        excess_pop = sumpop(period_data) .- pop_loc_target

        # Compute error
        moment_error = sim_moments - target_moments
        error_params = mean(abs.(get_moment_vector(moment_error)))
        error_rent = mean(x->x^2, excess_H_rent)
        error_pop = mean(x->x^2, excess_pop)
        error_total = error_params + error_rent + error_pop
        stepsize = next_stepsize!(scheduler, error_total, true)

        # Update guesses
        params = update_params(params, sim_moments, target_moments, stepsize)
        apply_price_change_effects!(loc_grid, stepsize, moment_error)
        ρ .= loc_grid.ρ
        @. ρ += excess_H_rent*stepsize
        @. α -= excess_pop*stepsize
        α .-= mean(α) .- 1

        # Report progress
        @show error_total
    end

    set_fundamentals_as_baseline_year!(period_data.loc_grid)
    return period_data
end

# Full Market Clearing #
#----------------------#
"Solve for steady-state rent ρ and real estate prices q, given fundamentals."
function solve_steady_state_rent_and_q!(
        period_data::PeriodData, params::Params, ρ=period_data.ρ, q=period_data.q;
        stepsize=1e-2, ε=2e-4
    )
    scheduler = JeffreysReallyBadScheduler(; stepsize, ε)

    while !scheduler.stop
        # Apply guesses
        period_data = solve_household_problem_steady_state!(period_data, params; ρ, q)
        validate_loc_grid(period_data)

        # Compute moments
        excess_H_rent = get_excess_H_rent(period_data, params)
        excess_H = get_excess_H_S(period_data, params)
        @show ρ[1]
        @show excess_H_rent[1]
        @show q[1]
        @show excess_H[1]

        # Compute errors
        error_rent = mean(x->x^2, excess_H_rent)
        error_H = mean(x->x^2, excess_H)
        error_total = error_rent + error_H
        stepsize = next_stepsize!(scheduler, error_total, true)

        # Update guesses
        @. ρ += excess_H_rent*stepsize
        @. q -= excess_H*stepsize
    end

    return period_data
end


################
# All Together #
################

function initialize_perioddata_for_calibration!(period_data::PeriodData, df::DataFrame, params::Params)
    (;loc_grid) = period_data

    # Structure location data
    loc_grid_empirical = read_empirical_steady_state(df)
    loc_target_moments, _ = read_empirical_target_moments(df)
    loc_grid .= get_calibration_loc_grid_init(loc_grid_empirical, loc_target_moments, params)

    return precompute!(period_data, params)
end
function initialize_perioddata_for_calibration!(period_data::PeriodData, params::Params; locunit="1990PUMA", year=1990, n_loc=N_LOC)
    df = read_empirical_steady_state_df(n_loc; locunit, year)
    return initialize_perioddata_for_calibration!(period_data, df, params)
end

function do_location_calibration!(
    period_data::PeriodData, df::DataFrame, params::Params;
    save=true, year=1990, stepsize=1e-4, ε=1e-4
)
    (;loc_grid) = period_data
    pop_loc_target = read_empirical_steady_state(df).pop
    initialize_perioddata_for_calibration!(period_data, df, params)
    
    calibrate_steady_state_rent_and_α!(period_data, params, pop_loc_target; stepsize, ε)
    save && write_steady_state_solution_baseline_year(loc_grid, year, "")
    return period_data
end
function do_location_calibration!(
    period_data::PeriodData, params::Params, n_loc::Int=N_LOC;
    locunit="1990PUMA", year=1990, kwargs...
)
    df = read_empirical_steady_state_df(n_loc; locunit, year)
    return do_location_calibration!(period_data, df, params; year, kwargs...)
end
do_location_calibration(params::Params; kwargs...) = do_location_calibration!(PeriodData(), params; kwargs...)

"Solve and save steady-state solution for a given climate change scenario."
function do_steady_state_solution!(
        period_data::PeriodData, n_dec::Int, RCP, params::Params, n_loc=N_LOC;
        locunit="1990PUMA", baseyear=1990, save=true, stepsize=1e-4, ε=1e-4,
    )
    (;loc_grid) = period_data
    loc_grid_baseyear = read_steady_state_solution_locgrid(baseyear, "", n_loc; locunit)
    ΔSST_path = get_ΔSST_path(n_dec, RCP)
    @. loc_grid = get_loc_climate(loc_grid_baseyear, ΔSST_path[end])

    # Solve for steady state
    sim_year = baseyear + (n_dec-1)*10
    period_data = precompute!(period_data, params)
    solve_steady_state_rent_and_q!(period_data, params, loc_grid.ρ, loc_grid.q; stepsize, ε)
    save && write_steady_state_solution(loc_grid, sim_year, RCP; locunit)

    return period_data
end

"""
Solve and save steady-state solution for a terminal temperature given by a branch off the path
TODO: Just index final steady-states by final SST, and don't worry about all this
RCP/year stuff.
1990PUMA_4.5C.csv
"""
function do_branched_steady_state_solution!(
        period_data::PeriodData, n_dec::Int, deci_branch, RCP_original, RCP_new, params::Params, n_loc=N_LOC;
        locunit="1990PUMA", baseyear=1990, save=true, stepsize=1e-4, ε=1e-4
    )
    (;loc_grid) = period_data
    loc_grid_baseyear = read_steady_state_solution_locgrid(baseyear, "", n_loc; locunit)

    ΔSST_path = get_ΔSST_surprise_path(n_dec, deci_branch, RCP_original, RCP_new)
    @. loc_grid = get_loc_climate(loc_grid_baseyear, ΔSST_path[end])

    # Solve for steady state
    sim_year = baseyear + (n_dec-1)*10
    period_data = precompute!(period_data, params)
    solve_steady_state_rent_and_q!(period_data, params, loc_grid.ρ, loc_grid.q; stepsize, ε)
    save && write_branched_ss_solution(loc_grid, RCP_new)

    return period_data
end


#################
# Read Solution #
#################

function read_steady_state_solution(sim_year::Int, RCP::String, params::Params, n_loc::Int=N_LOC; locunit="1990PUMA")
    loc_grid = read_steady_state_solution_locgrid(sim_year, RCP, n_loc; locunit)
    return solve_household_problem_steady_state(loc_grid, params)
end
read_steady_state_solution(params::Params) = read_steady_state_solution(1990, "", params)
