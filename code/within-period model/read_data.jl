
include("../../file_paths.jl")

indexdf(df, colname, value) = only(filter(row -> row[colname] == value, df))

load_empirical_wealth_dist() = CSV.read(WEALTH_DIST_PATH, DataFrame)
load_pairwise_distances(n_loc=N_LOC) = load(CLEANED_DISTANCE_PATH, "pairwise_distances")[1:n_loc, 1:n_loc]

"""
Generate an initial guess for spatial params, using the actual params in SPATIAL_PARAMS_PATH and
the empirical moments in SPATIAL_MOMENTS_2020_PATH.
"""
function read_spatial_params_initial_guess(;subset_locations=true)
    df_params = CSV.read(SPATIAL_PARAMS_PATH, DataFrame)
    df_moments = CSV.read(SPATIAL_MOMENTS_2020_PATH, DataFrame)

    @assert df_params.PUMA2020 == df_moments.PUMA2020

    GISMATCH = df_params.PUMA2020
    # Parameters
    elasticity = df_params.elasticity
    A_g = df_params.A_g
    α_g = df_params.α_g
    δ_g = df_params.delta_g
    H_S = df_moments.H_S ./ 1e6
    δ = df_moments.delta
    δ_bar = copy(δ)

    # Guesses
    A = df_moments.mean_labour_earnings ./ mean(df_moments.mean_labour_earnings)
    α = df_moments.population ./ mean(df_moments.population)
    q = df_moments.q ./ 1000
    Π = H_S./q.^elasticity
    A_bar = copy(A)
    α_bar = copy(α)
    
    n_loc = subset_locations ? N_LOC : length(GISMATCH)
    spatial_vec = StructArray([Location(;
        GISMATCH=GISMATCH[i],
        A=A[i],
        α=α[i],
        δ=δ[i],
        H_S=H_S[i],
        elasticity=elasticity[i],
        Π=Π[i],
        A_bar=A_bar[i],
        α_bar=α_bar[i],
        δ_bar=δ_bar[i],
        A_g=A_g[i],
        α_g=α_g[i],
        δ_g=δ_g[i]
    ) for i in 1:n_loc])

    return spatial = pad_dims(spatial_vec; left=3, ndims_new=4)
end

"Read 2020 prices."
function read_2020_prices(;subset_locations=true, steady_state=true)
    df = CSV.read(SPATIAL_MOMENTS_2020_PATH, DataFrame)

    GISMATCH = df.PUMA2020
    q = df.q ./ 1000
    ρ = df.mean_rent ./ 1000 .* 12 .* 10 # Decadal
    q_last = steady_state ? q : fill(NaN, length(GISMATCH))

    prices_vec = LocalPrices.(q, ρ, q_last) |> StructArray

    n_loc = subset_locations ? N_LOC : length(GISMATCH)
    return prices = pad_dims(prices_vec[1:n_loc]; left=3, ndims_new=4)
end

"Read 2020 empirical spatial moments."
function read_location_moments(subset_locations=true)
    df = CSV.read(SPATIAL_MOMENTS_2020_PATH, DataFrame)

    GISMATCH = df.PUMA2020
    pop = df.population ./ 1e6
    H = df.H_S ./ 1e6
    housing_price = df.q ./ 1000

    homeown_frac = df.homeownership_rate
    total_ownocc_valueh = df.total_owner_occupied_home_value ./ 1000 ./ 1e6
    mean_rent = df.mean_rent ./ 1000 .* 12 .* 10 # Decadal
    mean_earn = df.mean_labour_earnings ./ 1000 .* 10 # Decadal
    mean_earn_workage = df.mean_labour_earnings_workingage ./ 1000 .* 10 # Decadal
    mean_income = df.mean_income ./ 1000 .* 10 # Decadal
    mean_income_workage = df.mean_income_workingage ./ 1000 .* 10 # Decadal

    location_moments_vec = LocationMoments.(GISMATCH, pop, H, housing_price, homeown_frac, total_ownocc_valueh, mean_rent, mean_earn, mean_earn_workage, mean_income, mean_income_workage) |> StructArray

    n_loc = subset_locations ? N_LOC : length(GISMATCH)
    return location_moments = pad_dims(location_moments_vec[1:n_loc]; left=3, ndims_new=4)
end

"Read 2020 empirical aggregate moments"
function read_aggregate_moments(moment=:all)
    df = CSV.read(AGGREGATE_MOMENTS_PATH, DataFrame)
    moment == :all && return df
    
    return indexdf(df, :moment, moment).value
end
