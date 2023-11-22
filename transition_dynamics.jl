
"""
Compute transition dynamics in the perfect-foresight model.

1. Define a sequence of location fundamentals over time.
2. Guess a sequence of prices over time.
3. Solve for a single age group, given a sequence of location values.
4. Solve for all age groups and store the solution.
5. Simulate forward from the initial distribution. (populations and housing stock)
6. Compute market clearing conditions.
7. Update price guesses.

Notes:
- We now need to think about housing prices.
  - Supply: (housing prices, old housing stock) -> housing stock
  - Demand: housing prices -> housing demand
  - Market clearing: housing stock - housing demand = 0
"""

######################
# Steady State Setup #
######################

# Construct a LocGridPath from Boundary LocGrids #
#------------------------------------------------#
"Copy equilibrium prices and quantities from loc_grid_ss into loc_grid_path[deci]."
function fill_in_period_equilibrium!(loc_grid_path::LocGridPath, loc_grid_ss::LocGrid, deci::Int)
    lgp_deci = LocGrid(loc_grid_path, deci)
    for var in [:q, :ρ, :pop, :H_D, :H, :q_last]
        getproperty(lgp_deci, var) .= getproperty(loc_grid_ss, var)
    end
    @assert all(zip(lgp_deci, loc_grid_ss)) do (x, y)
        #all(getfield(x, s)==getfield(y, s) for s=[:Π, :H_bar, :elasticity, :A_bar, :α_bar, :δ_bar, :A_g, :α_g, :δ_g])
        x == y
    end
    return loc_grid_path
end

"Guess intermediate prices by linearly interpolating between the start and end steady states."
function interpolate_intermediate_prices!(loc_grid_path::LocGridPath)
    n_dec = size(loc_grid_path, DEC_DIM)
    # Interpolate prices linearly between the start and end steady states
    for var in [:q, :ρ]
        for loci = 1:N_LOC
            loc_path = selectdim(loc_grid_path, LOC_DIM, loci)
            var_start = first(getproperty(loc_path, var))
            var_end = last(getproperty(loc_path, var)[end])
            vec(getproperty(loc_path, var)) .= range(var_start, var_end; length=n_dec)
        end
    end
    return loc_grid_path
end

function LocGridPath(loc_grid_start::LocGrid, ΔSST_path)
    loc_grid_path = get_loc_climate.(loc_grid_start, ΔSST_path)
    for var in [:q, :ρ, :pop, :H_D, :H]
        getproperty(loc_grid_path, var) .= NaN
    end

    fill_in_period_equilibrium!(loc_grid_path, loc_grid_start, 1)
    return loc_grid_path
end

"""
    Compute a path of local climate conditions and equilibrium prices from an initial
    steady state, a final steady state, and a path of global temperatures.
"""
function LocGridPath(loc_grid_start::LocGrid, loc_grid_end::LocGrid, ΔSST_path; guess_intermediate_prices=true)
    loc_grid_path = LocGridPath(loc_grid_start, ΔSST_path)

    fill_in_period_equilibrium!(loc_grid_path, loc_grid_end, get_n_dec(loc_grid_path))

    guess_intermediate_prices && return interpolate_intermediate_prices!(loc_grid_path)
    return loc_grid_path
end

function read_loc_grid_path_from_boundaries(n_dec, RCP; baseyear=1990, locunit="1990PUMA")
    endyear = baseyear + (n_dec-1)*10

    loc_grid_start = read_steady_state_solution_locgrid(baseyear, ""; locunit)
    loc_grid_end = read_steady_state_solution_locgrid(endyear, RCP; locunit)
    ΔSST_path = get_ΔSST_path(n_dec, RCP)

    return LocGridPath(loc_grid_start, loc_grid_end, ΔSST_path)
end

function read_surprise_loc_grid_path_from_boundaries(start_year, end_year, sim_year, RCP_original, RCP_new; locunit="1990PUMA")
    loc_grid_mid = read_midtransition_solution(start_year, end_year, sim_year, RCP_original; locunit)
    loc_grid_end = read_branched_ss_solution(RCP_new)
    deci_sim = (sim_year - start_year)/10 + 1 |> Int
    n_dec = (end_year - start_year)/10 + 1 |> Int

    ΔSST_path = get_ΔSST_surprise_path(n_dec, deci_sim, RCP_original, RCP_new)
    return LocGridPath(loc_grid_mid, loc_grid_end, ΔSST_path)
end

# Generate a PathData object from Boundary Steady State LocGrids #
#----------------------------------------------------------------#
function solve_boundary_steady_states!(path_data::PathData, params::Params)
    (;n_dec) = path_data
    solve_household_problem_steady_state!(PeriodData(path_data, 1), params)
    solve_household_problem_steady_state!(PeriodData(path_data, n_dec), params)
    return path_data
end

function get_path_data_from_initial_steady_state(params::Params, n_dec, start_year::Int=1990; n_loc::Int=N_LOC)
    loc_grid = read_steady_state_solution_locgrid(start_year, "", n_loc)
    ΔSST_path = pad_dims(FLOAT_PRECISION[0, fill(NaN, n_dec-1)...], DEC_DIM-1, 0)
    loc_grid_path = LocGridPath(loc_grid, ΔSST_path)
    path_data = PathData(loc_grid_path)
    solve_household_problem_steady_state!(PeriodData(path_data, 1), params)
    return path_data
end

"""
    Get a PathData object from two steady state solutions.
    The terminal value function V is the steady value function of the terminal steady state.
    The initial state distribution λ is the terminal state distribution of the terminal steady state.
    
    Note that I am not yet storing the steady state V or λ. Instead, I assume that the stored
    prices represent the steady state price, then reconstruct V and λ.
"""
function get_path_data_with_boundary_steady_states(n_dec, RCP, params; baseyear=1990, locunit="1990PUMA")::PathData
    loc_grid_path = read_loc_grid_path_from_boundaries(n_dec, RCP; baseyear, locunit)
    path_data = preallocate_and_precompute(loc_grid_path, params)
    solve_boundary_steady_states!(path_data, params)
    return path_data
end

function get_surprise_path_data(start_year, end_year, sim_year, RCP_original, RCP_new; locunit="1990PUMA")
    loc_grid_path = read_surprise_loc_grid_path_from_boundaries(start_year, end_year, sim_year, RCP_original, RCP_new; locunit)
    path_data = preallocate_and_precompute(loc_grid_path, params)
    solve_boundary_steady_states!(path_data, params)
    return path_data
end


########################
# Household's Probelem #
########################

"Solve V backwards for an age group agei who either dies or goes into the new steady state in period deci."
function solve_V_backward_transition_dec!(path_data::PathData, deci::Int, agei_end::Int, params::Params)
    (;loc_grid_path) = path_data
    global deci_last = deci
    n_dec = size(loc_grid_path, DEC_DIM)

    age_data = AgeData(path_data, agei_end, deci)#; age_solution_deci)
    loc_grid = LocGrid(loc_grid_path, deci)

    # Compute the bequest utility for a household who dies in this period
    # We can start with a non-dying household iff they continue into the new steady state (yea and they shall be saved)
    if agei_end == N_AGE
        age_data.V_next .= get_V_bequest!(age_data, loc_grid, params)
    else
        @assert deci == n_dec
    end

    for agei in reverse(1:agei_end)
        global agei_last = agei

        solve_period!(age_data, age_data.V_next, params)
        
        # If we have reached birth or the initial steady state, break
        min(agei, deci) == 1 && break

        V_start = age_data.V_price
        deci -= 1
        age_data = AgeData(path_data, agei-1, deci)#; age_solution_deci)
        age_data.V_next .= V_start
    end
    return path_data
end

"Solve V backwards along the transition path, using either new steady state or bequest continuation values."
function solve_V_backward_transition!(path_data::PathData, params::Params)
    (;n_dec) = path_data

    # Solve for age groups who continue on into the new steady state (yea and they shall be saved)
    #Threads.@threads
    for agei in 1:N_AGE-1
        solve_V_backward_transition_dec!(path_data, n_dec, agei, params)
    end

    # Solve for age groups who die during the transition
    #Threads.@threads
    for dec in reverse(1:n_dec)
        solve_V_backward_transition_dec!(path_data, dec, N_AGE, params)
    end
    return path_data
end

function solve_λ_forward_transition_dec!(path_data::PathData, deci::Int, agei_start::Int, params::Params)
    (;loc_grid_path) = path_data
    global deci_last = deci
    n_dec = size(loc_grid_path, DEC_DIM)

    # If they are born mid-transition, use the cache from when they are born
    # If they are born before the transition, use the cache from when they die
    #age_solution_deci = agei_start == 1 ? deci : N_AGE - agei_start + 1
    age_data = AgeData(path_data, agei_start, deci)#; age_solution_deci)

    # We can start with a non-birth household only if they are coming from the old steady state    
    if agei_start == 1
        λ_start = get_λ_init_steady_state(PeriodData(path_data, deci), params)
    else
        @assert deci == 1
        λ_start = age_data.λ_start
    end

    # Simulate forward
    for agei in agei_start:N_AGE
        global agei_last = agei
        λ_start = iterate_λ(λ_start, age_data, params)

        # If we reach the end of life or the transition, break
        (deci == n_dec || agei==N_AGE) && break

        deci += 1
        global deci_last = deci
        age_data = AgeData(path_data, agei+1, deci)#; age_solution_deci)
    end

    return path_data
end

function solve_forward_transition!(path_data::PathData, params::Params)
    (;n_dec) = path_data

    # Solve for H
    H_last = path_data.H[1,1,1,1,:,1,1,1]
    for deci in 1:n_dec
        path_data.H[1,1,1,1,:,1,1,deci] .= H_last
        simulate_H_forward!(PeriodData(path_data, deci), params)
        H_last .= path_data.H[1,1,1,1,:,1,1,deci]
    end

    # Solve for age groups who are born before the transition begins
    for agei in 2:N_AGE
        solve_λ_forward_transition_dec!(path_data, 1, agei, params)
    end
    # Solve for age groups who are born during the transition
    for deci in 1:n_dec
        solve_λ_forward_transition_dec!(path_data, deci, 1, params)
    end

    # Fill in loc_grid with new equilibria
    write_equilibrium_to_loc_grid!(path_data)

    return path_data
end

function update_params!(pd::PathData, param_updates)
    (;loc_grid_path) = pd
    for s in keys(param_updates)
        if hasfield(Location, s)
            getproperty(loc_grid_path, s) .= param_updates[s]
        else
            error("Unknown field $s")
        end
    end
end

function fill_in_q_last!(loc_grid_path::LocGridPath)
    n_dec = size(loc_grid_path, DEC_DIM)
    # Fill in q_last
    for deci in 2:n_dec
        loc_grid = LocGrid(loc_grid_path, deci)
        loc_grid_last = LocGrid(loc_grid_path, deci-1)
        loc_grid.q_last .= loc_grid_last.q
    end
    return loc_grid_path
end

function solve_household_problem_transition!(
        path_data::PathData, params::Params; param_updates...
    )
    update_params!(path_data, param_updates)
    fill_in_q_last!(path_data.loc_grid_path)
    precompute!(path_data, params)
    solve_V_backward_transition!(path_data, params)
    solve_forward_transition!(path_data, params)
    return path_data
end

###################
# Market Clearing #
###################

function get_excess_H_rent(path_data::PathData, params::Params)
    excess_H_rent = get_excess_H_rent.(PeriodDataPath(path_data), params)
    return stack_locdata(excess_H_rent)
end

function get_excess_H_S(path_data::PathData, params::Params)
    excess_H = get_excess_H_S.(PeriodDataPath(path_data), params)
    return stack_locdata(excess_H)
end


###################
# Transition Path #
###################

function solve_transition_rent_and_q!(
        path_data::PathData, params::Params, ρ_init=path_data.ρ, q_init=path_data.q;
        stepsize=1e-2, ε=1e-4
    )
    ρ, q = deepcopy.((ρ_init, q_init))
    scheduler = JeffreysReallyBadScheduler(; stepsize, ε)

    while !scheduler.stop
        # Apply guesses
        solve_household_problem_transition!(path_data, params; ρ, q)
        validate_loc_grid_path(path_data)

        # Compute moments
        excess_H_rent = get_excess_H_rent(path_data, params)
        excess_H_D = -get_excess_H_S(path_data, params)

        # Compute errors
        error_rent = mean(x->x^2, excess_H_rent)
        error_H_D = mean(x->x^2, excess_H_D)
        error_total = error_rent + error_H_D
        stepsize = next_stepsize!(scheduler, error_total, true)
        
        # Update guesses
        @. ρ += excess_H_rent*stepsize
        @. q += excess_H_D*stepsize
    end

    return path_data
end


################
# All Together #
################

"""
The concept of start_year is different from baseyear.
baseyear is the year from which climate changes are measured.
start_year is just the beginning of a given simulation.
"""
function save_midtransition_equilibrium(path_data::PathData, middle_year, start_year, end_year, RCP; locunit="1990PUMA")
    deci = div(middle_year - start_year, 10) + 1
    loc_grid_middle = LocGrid(path_data.loc_grid_path, deci)
    write_midtransition_solution(loc_grid_middle, start_year, end_year, middle_year, RCP; locunit)
end

function do_transition_solution!(
        path_data::PathData, n_dec::Int, RCP::String, params::Params;
        baseyear=1990, saveyear=2050, locunit="1990PUMA",
        stepsize=1e-4, ε=2e-4
    )
    endyear = baseyear + (n_dec-1)*10

    solve_transition_rent_and_q!(path_data, params; stepsize, ε)
    save_midtransition_equilibrium(path_data, saveyear, baseyear, endyear, RCP)
    write_transition_panel(path_data.loc_grid_path, baseyear, endyear, RCP)
    return path_data
end
function do_transition_solution(
        n_dec::Int, RCP::String, params::Params;
        baseyear=1990, locunit="1990PUMA", kwargs...
    )
    path_data = get_path_data_with_boundary_steady_states(n_dec, RCP, params; baseyear, locunit)
    return do_transition_solution!(path_data, n_dec, RCP, params; baseyear, locunit, kwargs...)
end

function do_surprise_transition_solution(
        start_year, end_year, sim_year, RCP_original, RCP_new, params;
        locunit="1990PUMA", stepsize=1e-4, ε=1e-4
    )
    path_data = get_surprise_path_data(start_year, end_year, sim_year, RCP_original, RCP_new; locunit)
    solve_transition_rent_and_q!(path_data, params; stepsize, ε)
    write_surprise_transition_panel(path_data.loc_grid_path, start_year, end_year, sim_year, RCP_original, RCP_new)
    return path_data
end
