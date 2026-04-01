
# Read #
#------#
get_test_params() = load_spatially_calibrated_prices_params().params
get_test_prices() = load_spatially_calibrated_prices_params().prices

function get_test_params_pd()
    (;prices, params) = load_spatially_calibrated_prices_params()
    pd = PeriodData(prices, params)
    return (;pd, params)
end

function get_test_pd(params::Params=get_test_params(), prices_init=get_test_prices(); q=nothing, ρ=nothing, q_last=nothing)
    return PeriodData(prices_init, params; q, ρ, q_last)
end

# Write #
#-------#
function get_dataframe_slices(pd::PeriodData, params::Params)
    year = 2010 + params.decade*10

    PUMA2020 = params.GISMATCH[:]
    q = pd.q[:]
    q_last = pd.q_last[:]
    rent = pd.ρ[:]
    H = params.H_S[:]
    population = get_pops(pd)[:]

    mean_wealth_agg = weightedmean(WEALTH_GRID, sum(pd.λ_start))
    mean_V_agg = weightedmean(stack(pd.V_start), stack(pd.λ_start))
    mean_V_birth_agg = weightedmean(pd.V_start[1], pd.λ_start[1])
    wealth_disp_agg = weightedmean((WEALTH_GRID .- mean_wealth_agg).^2, sum(pd.λ_start))
    V_disp_agg = weightedmean((stack(pd.V_start) .- mean_V_agg).^2, stack(pd.λ_start))
    V_birth_disp_agg = weightedmean((pd.V_start[1] .- mean_V_birth_agg).^2, pd.λ_start[1])

    mean_wealth = weightedmean(WEALTH_GRID, sum(pd.λ_start); dims=(K_DIM, Z_DIM, H_DIM))[:]
    mean_V = weightedmean(stack(pd.V_start), stack(pd.λ_start); dims=(K_DIM, Z_DIM, H_DIM, AGE_DIM))[:]

    mean_wealth_byage = weightedmean(WEALTH_GRID, stack(pd.λ_start); dims=(K_DIM, Z_DIM, H_DIM))
    mean_V_byage = weightedmean(stack(pd.V_start), stack(pd.λ_start); dims=(K_DIM, Z_DIM, H_DIM))

    mean_wealth_age1 = mean_wealth_byage[1,1,1,:,1]
    mean_wealth_age2 = mean_wealth_byage[1,1,1,:,2]
    mean_wealth_age3 = mean_wealth_byage[1,1,1,:,3]
    mean_wealth_age4 = mean_wealth_byage[1,1,1,:,4]
    mean_wealth_age5 = mean_wealth_byage[1,1,1,:,5]
    mean_wealth_age6 = mean_wealth_byage[1,1,1,:,6]

    mean_V_age1 = mean_V_byage[1,1,1,:,1]
    mean_V_age2 = mean_V_byage[1,1,1,:,2]
    mean_V_age3 = mean_V_byage[1,1,1,:,3]
    mean_V_age4 = mean_V_byage[1,1,1,:,4]
    mean_V_age5 = mean_V_byage[1,1,1,:,5]
    mean_V_age6 = mean_V_byage[1,1,1,:,6]

    mean_V_bywealth = weightedmean(stack(pd.V_start), stack(pd.λ_start); dims=(Z_DIM, H_DIM, AGE_DIM))

    mean_V_birth_belowmed = weightedmean(pd.V_start[1][1:36,:,:,:], pd.λ_start[1][1:36,:,:,:]; dims=(K_DIM, Z_DIM, H_DIM))[:]
    mean_V_birth_abovemed = weightedmean(pd.V_start[1][37:end,:,:,:], pd.λ_start[1][37:end,:,:,:]; dims=(K_DIM, Z_DIM, H_DIM))[:]

    mean_gain = weightedmean((pd.q-pd.q_last).*H_GRID, sum(pd.λ_start); dims=(K_DIM, Z_DIM, H_DIM))[:]

    df_panelslice = DataFrame(;year, PUMA2020, q, q_last, rent, H, population, mean_wealth, mean_V,
        mean_wealth_age1, mean_wealth_age2, mean_wealth_age3, mean_wealth_age4, mean_wealth_age5, mean_wealth_age6,
        mean_V_age1, mean_V_age2, mean_V_age3, mean_V_age4, mean_V_age5, mean_V_age6,
        mean_V_birth_belowmed, mean_V_birth_abovemed,
        mean_gain
    )

    df_seriesslice = DataFrame(;year, mean_wealth_agg, mean_V_agg, mean_V_birth_agg, wealth_disp_agg, V_disp_agg, V_birth_disp_agg)
    return (;df_panelslice, df_seriesslice)
end

#=
write_csv(dir, df, year=2020) = CSV.write(dir*string(year)*".csv", df)
read_csv(dir, year=2020) = CSV.File(dir*string(year)*".csv") |> DataFrame

"Save steady state solution to file."
function write_steady_state_solution(loc_grid::LocGrid, sim_year::Int)
    n_loc = size(loc_grid, LOC_DIM)
    return write_csv(SS_SOLN_DIRPATH*n_loc*"loc_", DataFrame(loc_grid), sim_year)
end
=#
