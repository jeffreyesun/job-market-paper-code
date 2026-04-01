
get_MIT_shock_params_path(T::Int, params_pre, params_post) = [params_pre, [deepcopy(params_post) for t=2:T]...]

# Counterfactual Parameters #
#---------------------------#
function get_α_shock_counterfactual_params(params_pre, T::Int)
    params_pre = deepcopy(params_pre)
    params_post = deepcopy(params_pre)
    params_post.spatial.α_bar[1] *= 1.5

    return get_MIT_shock_params_path(T, params_pre, params_post)
end

function get_climate_change_counterfactual_params(params_pre, T::Int; δ_factor=1.0)
    params_pre = deepcopy(params_pre)
    pp = [deepcopy(params_pre) for t=1:T]
    for t=1:T
        pp[t].decade = t
        pp[t].scenario_ind = SCENARIO_IND_INIT
    end

    for t=2:T
        pp[t].spatial.δ_bar .*= δ_factor
    end
    return pp
end

function get_Π_treated_distances(params)
    # Get the set of ``high-risk'' locations where flood risk exceeds 95th percentile
    δ_95pc = quantile(params.δ[:], 0.95)
    high_risk_loc_mask = params.δ .>= δ_95pc

    # Compute distance of each location with the nearest high-risk location
    distances = load_pairwise_distances()
    distances_to_high_risk = [minimum(distances[loci, high_risk_loc_mask[:]]) for loci=1:N_LOC]

    # Find 5% of locations closest to high-risk locations
    high_risk_distance_10pc = quantile(distances_to_high_risk, 0.10)
    treated_loc_mask = [(distances_to_high_risk[loci] <= high_risk_distance_10pc) && !high_risk_loc_mask[loci] for loci=1:N_LOC]

    return treated_loc_mask
end

function get_climate_change_Π_policy_counterfactual_params(params_pre, T::Int; δ_factor=1.0, treated_Π_factor=1.05)
    params_pre = deepcopy(params_pre)
    pp = [deepcopy(params_pre) for t=1:T]
    for t=1:T
        pp[t].decade = t
        pp[t].scenario_ind = SCENARIO_IND_INIT
    end

    treated_loc_mask = get_Π_treated_distances(params_pre)

    for t=2:T
        pp[t].Π[treated_loc_mask] .*= treated_Π_factor
    end

    for t=2:T
        pp[t].spatial.δ_bar .*= δ_factor
    end
    return pp
end

function get_migration_ease_counterfactual_params(params_old; F_u_fixed_factor=0.5, F_u_dist_factor=0.5)
    params = deepcopy(params_old)
    params.F_u_fixed *= F_u_fixed_factor
    params.F_u_dist *= F_u_dist_factor
    return params
end

# Counterfactual Solutions #
#--------------------------#
function solve_α_shock_transition(params_pre; verbosity=2, T=20)
    pp = get_α_shock_counterfactual_params(params_pre, T)
    return (;td, pp) = solve_transition(pp; verbosity)
end

function solve_climate_change_transition(params_pre; verbosity=2, T=20, δ_factor=1.0)
    pp = get_climate_change_counterfactual_params(params_pre, T; δ_factor)
    return (;td, pp) = solve_transition(pp; verbosity)
end

function solve_climate_change_Π_policy_transition(params_pre; verbosity=2, T=20, δ_factor=1.0, treated_Π_factor=1.05)
    pp = get_climate_change_Π_policy_counterfactual_params(params_pre, T; δ_factor, treated_Π_factor)
    return (;td, pp) = solve_transition(pp; verbosity)
end
