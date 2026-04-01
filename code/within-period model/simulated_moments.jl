
# Market Clearing Conditions #
#----------------------------#
# Housing Asset Demand
get_H_D(pd) = sum(sum(λ.*get_h_grid(λ); dims=(K_DIM,Z_DIM,H_DIM)) for λ=pd.λ_end)

# Housing Consumption Demand
function _get_H_rent_age(wealthi_postc_k_prec_age, h_share_byloc, λ_prec_age, ρ)

    wealth_grid = get_wealth_grid(wealthi_postc_k_prec_age)
    spend = wealth_grid .- wealth_grid[wealthi_postc_k_prec_age]
    inner = h_share_byloc .* spend .* λ_prec_age
    return H_rent_age = sum(inner; dims=(K_DIM,Z_DIM,H_DIM))./ρ
end

function get_H_rent(data::Union{AgeData,AgeDataGPU}, params; ages=1:N_AGE)
    (;wealthi_postc_k_prec, prices) = data
    λ_prec = data.λ_postincome
    (;ρ) = prices
    (;γ, σ) = params

    # Precompute housing share of total consumption spending by location
    h_share_byloc = get_h_demand.(1, ρ, γ, σ).*ρ

    return H_rent = _get_H_rent_age(wealthi_postc_k_prec, h_share_byloc, λ_prec, ρ)
end

function get_H_rent(data::Union{PeriodData, PeriodDataGPU}, params; ages=1:N_AGE)
    (;wealthi_postc_k_prec, prices) = data
    λ_prec = data.λ_postincome
    (;ρ) = prices
    (;γ, σ) = params

    # Precompute housing share of total consumption spending by location
    h_share_byloc = get_h_demand.(1, ρ, γ, σ).*ρ

    return H_rent = sum(_get_H_rent_age(wealthi_postc_k_prec[agei], h_share_byloc, λ_prec[agei], ρ) for agei=ages)
end

# Spatial Calibration Targets #
#-----------------------------#
function get_mean_incomes(pd::PeriodData, params::Params; dims=(K_DIM, Z_DIM, H_DIM), ages=1:N_AGE)
    incomes = []
    λ_postbuy_sums = []
    for agei=ages
        iterate_λ_forward!(pd[agei], pd.λ_start[agei], params)
        all_incomes = get_income.(WEALTH_GRID, Z_GRID[agei], H_GRID, params.spatial, pd.prices, params)
        push!(incomes, sum(all_incomes .* pd[agei].λ_postbuy; dims))
        push!(λ_postbuy_sums, sum(pd[agei].λ_postbuy; dims))
    end
    return sum(incomes) ./ sum(λ_postbuy_sums)
end

# Population
get_pops(pd::Union{PeriodData, PeriodDataGPU}; ages=1:N_AGE) = sum(sum(λ; dims=(K_DIM,Z_DIM,H_DIM)) for λ=pd.λ_end[ages])

get_pops(ad::AgeData) = sum(ad.λ_end; dims=(K_DIM,Z_DIM,H_DIM))

# Earnings
function get_tot_earnings(pd::Union{PeriodData, PeriodDataGPU}, params; ages=1:N_AGE)
    (;A) = params.spatial
    (;λ_start) = pd
    return sum(sum(λ_start[agei].*get_z_grid(agei,A).*A; dims=(K_DIM,Z_DIM,H_DIM)) for agei=ages)
end

get_mean_earnings(pd::Union{PeriodData,PeriodDataGPU}, params; ages=1:N_AGE) = get_tot_earnings(pd, params; ages) ./ get_pops(pd; ages)

# Non-Spatial Calibration Targets #
#---------------------------------#
# Rent to Earnings Ratio
get_rent_to_earnings_workage(pd::Union{PeriodData, PeriodDataGPU}, params) = get_H_rent(pd, params; ages=1:4) .* pd.ρ ./ get_tot_earnings(pd, params; ages=1:4)
get_average_rent_to_earnings_workage(pd::Union{PeriodData, PeriodDataGPU}, params) = weightedmean(get_rent_to_earnings_workage(pd, params), get_pops(pd))

"Get relative mean wealth of 70-year-olds to 60-year-olds."
function get_old_relative_meanwealth(pd)
    meanwealth_70 = weightedmean(WEALTH_GRID, AgeData(pd, N_AGE-1).λ_end)
    meanwealth_60 = weightedmean(WEALTH_GRID, AgeData(pd, N_AGE-2).λ_end)
    return old_relative_meanwealth = meanwealth_70 / meanwealth_60
end

"Get ratio of price to decadal rent by location."
get_price_to_rent(pd) = pd.q./pd.ρ

"Get population-weighted average ratio of price to decadal rent."
get_average_price_to_rent(pd) = weightedmean(get_price_to_rent(pd), get_pops(pd))

## Aggregate Moving Flows
"Get aggregate moving flows."
function get_agg_moving_flows(ad::AgeData)
    # Solve so that cached values of intermediate calculations are populated correctly
    iterate_V_backward!(ad, ad.V_end, params)
    iterate_λ_forward!(ad, ad.λ_start, params)

    
    if ACTIVELY_DISPATCH_TO_GPU
        λ_premove = cu(ad.λ_postsell)
        eψV_move_tilde = cu(ad.eψV_move_tilde)
        eψFu_inv = cu(ad.eψFu_inv)
        eψV_postmove_tilde = cu(ad.eψV_postmove_tilde)
        origin_weights = cu(ad.origin_weights)
        agg_flows_mat = CUDA.zeros(N_LOC, N_LOC)
    else
        λ_premove = ad.λ_postsell
        (;eψV_move_tilde, eψFu_inv, eψV_postmove_tilde, origin_weights) = ad
        agg_flows_mat = zeros(FLOAT, N_LOC, N_LOC)
    end

    # Get matrices of conditional pre- and post-move values
    eψV_postmove_tilde_mat = reshape(eψV_postmove_tilde, (N_K*N_Z, N_LOC))
    @. origin_weights = λ_premove[:,:,1:1,:] / eψV_move_tilde
    origin_mat = reshape(origin_weights, (N_K*N_Z, N_LOC))

    # Sum flow matrices for each k, z
    for i in 1:N_K*N_Z
        flows_mat = origin_mat[i,:] .* eψFu_inv
        flows_mat .*= eψV_postmove_tilde_mat[i:i,:]
        agg_flows_mat .+= flows_mat
    end

    # Check conservation of mass
    #@assert sum(agg_flows_mat) ≈ sum(λ_premove[:,:,1:1,:])
    #@assert sum(agg_flows_mat; dims=2)' ≈ sum(reshape(λ_premove[:,:,1:1,:], (N_K*N_Z, N_LOC)); dims=1)
    #@assert sum(agg_flows_mat; dims=1) ≈ sum(reshape(ad.λ_postmove[:,:,1:1,:], (N_K*N_Z, N_LOC)); dims=1)

    return cpu(agg_flows_mat)
end
get_agg_moving_flows(pd::PeriodData) = sum(get_agg_moving_flows(pd[agei]) for agei in 1:N_AGE)

"Get share of population moving."
function get_share_moving(pd::PeriodData)
    agg_flows_mat = get_agg_moving_flows(pd)
    agg_flows_mat .*= (1 .- I(N_LOC)) # Zero out diagonal
    #@assert sum(get_pops(pd)) ≈ POP_SUM
    return moving_share = sum(agg_flows_mat) / sum(get_pops(pd))
end
"Alternative computation of the same thing."
function get_share_moving_alternative(pd::PeriodData)
    tot_pop_stay = FLOAT(0.0)

    for agei=1:N_AGE
        ad = pd[agei]    
        iterate_V_backward!(ad, ad.V_end, params)
        iterate_λ_forward!(ad, ad.λ_start, params)

        (;origin_weights, eψFu_inv, λ_postmove, eψV_postmove_tilde) = ad

        #@. origin_weights = λ_premove[:,:,1:1,:] / eψV_move_tilde
        origin_mat = reshape(origin_weights, N_K*N_Z, N_LOC)
        λ_stay = origin_mat .* diag(eψFu_inv)' .* reshape(eψV_postmove_tilde, N_K*N_Z, N_LOC)
        tot_pop_stay += sum(λ_stay) + sum(@view λ_postmove[:,:,2:end,:])
    end

    return 1 - tot_pop_stay / POP_SUM
end

"Average distance of a move over 20 miles within the contiguous United States."
function get_avg_mig_distance(pd)
    agg_flows_mat = get_agg_moving_flows(pd)
    mask = vec(DIST_LOC) .> 20

    sorted_indices = sortperm(DIST_LOC[mask])
    v_sorted = DIST_LOC[mask][sorted_indices]
    w_sorted = agg_flows_mat[mask][sorted_indices]
    
    median_plus_idx = findfirst(cumsum(w_sorted) .>= sum(w_sorted)/2)
    median_minus_idx = findlast(cumsum(w_sorted) .<= sum(w_sorted)/2)
    w_minus = w_sorted[median_minus_idx]
    w_plus = w_sorted[median_plus_idx]

    return (v_sorted[median_minus_idx]*w_minus + v_sorted[median_plus_idx]*w_plus)/(w_minus + w_plus)
end

"Earnings Dispersion (Working Age)"
function get_earnings_dispersion_workage(pd, params)
    λ_start = stack(pd.λ_start[1:4])
    z_grid_stack = stack(Z_GRID[1:4])
    
    logearn = log.(z_grid_stack.*params.A)
    return var_logearn = sum(λ_start.*logearn.^2)/sum(λ_start) - (sum(λ_start.*logearn)/sum(λ_start))^2
end

"Probability of Selling Home"
function get_average_P_sell(pd, params)
    mass_sell = FLOAT(0)

    for agei=1:N_AGE
        ad = pd[agei]
        iterate_V_backward!(ad, ad.V_end, params)
        iterate_λ_forward!(ad, ad.λ_start, params)

        mass_sell += vec(ad.P_sell)⋅vec(ad.λ_postprice)
    end

    return mass_sell / POP_SUM
end

# 2SLS Estimate of Housing-Goods Elasticity
"Get 2SLS point estimate of housing-consumption elasticity"
function get_eos_coef(df::DataFrame)
        
    global eos_res1 = lm(@formula(log(rent_level) ~ elasticity + log(mean_earn_workage)), df)
    coef_elas = coef(eos_res1)[2]
    
    global eos_res2 = lm(@formula(log(H_D_percap/income_after_rent) ~ elasticity), df[df.income_after_rent .> 0,:])
    coef_elas_H_D = coef(eos_res2)[2]
    
    return eos_hat = coef_elas_H_D/coef_elas
end

function get_eos_coef_model(pd, params)
    mean_earn = get_mean_earnings(pd, params; ages=1:4)
    H_D_percap = params.spatial.H_S ./ get_pops(pd)

    df = DataFrame(;
        PUMA2020 = params.GISMATCH[:],
        rent_level = Float64.(pd.ρ[:]),
        mean_earn_workage = Float64.(mean_earn[:]),
        elasticity = Float64.(params.elasticity[:]),
        H_D_percap = Float64.(H_D_percap[:]),
        income_after_rent = Float64.(mean_earn .- pd.ρ.*H_D_percap)[:],
    )
    return eos_hat = get_eos_coef(df)
end

"OLS Estimate of Price Elasticity of Demand for Residence"
function get_rent_wage_coef(df::DataFrame)
    # Ideally, temperature variables
    #res = lm(@formula(log(mean_rent) ~ log(mean_labour_earnings_workingage) + delta + elasticity + HDD_g + CDD_g), df)
    global rent_wage_res = lm(@formula(log(rent_level) ~ log(mean_labour_earnings_workingage) + delta + elasticity + HDD_g + CDD_g), df)
    #global rent_wage_res = lm(@formula(log(rent_level) ~ log(mean_labour_earnings_workingage)), df)
    return coef(rent_wage_res)[2]
end

function get_rent_wage_coef_model(pd, params; use_q_instead=false)
    solve_household_problem_steady_state!(pd, params)
    
    data_moments = read_location_moments()
    df_params = CSV.read(SPATIAL_PARAMS_PATH, DataFrame)[1:N_LOC, :]
    
    df = DataFrame(;
        PUMA2020 = params.GISMATCH[:],
        rent_level = use_q_instead ? pd.q[:] : pd.ρ[:],
        mean_labour_earnings_workingage = data_moments.mean_earn_workage[:],
        delta = params.δ[:],
        elasticity = params.elasticity[:],
        HDD_g = df_params.HDD_g[:],
        CDD_g = df_params.CDD_g[:],
    )
    for col=names(df)[2:end]
        df[!, col] = Float64.(df[!, col])
    end

    return get_rent_wage_coef(df)
end

# Untargeted Moments #
#--------------------#
get_mean_wealth(pd::PeriodData) = weightedmean(WEALTH_GRID, pd.λ_end)