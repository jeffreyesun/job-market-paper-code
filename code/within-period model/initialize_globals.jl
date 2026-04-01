
const YOUNG_WEALTH_INIT = zeros(FLOAT, N_K)
const YOUNG_Z_FLAT = zeros(FLOAT, N_Z)
const DIST_LOC = zeros(FLOAT, N_LOC, N_LOC)

# Spatial Data #
#--------------#
function setglobal_distloc()
    global DIST_LOC .= FLOAT.(load_pairwise_distances())
    return nothing
end

# Wealth Distribution #
#---------------------#
function get_wealth_dist(quartile_lens, Q3_i, tail_width; n_empty_upper_cells=20)
    wealth_dist_init = Float32.(vcat([fill(1/len_Q/4, len_Q) for len_Q in quartile_lens]..., zeros(n_empty_upper_cells+1)))
    tail_grid = range(-1, 1, length=N_K-Q3_i-n_empty_upper_cells)
    @views wealth_dist_init[Q3_i:N_K-n_empty_upper_cells-1] .-= tail_width .* tail_grid
    return wealth_dist_init
end

function get_wealth_dist_matching_mean(quartile_lens, Q3_i, mean_wealth; rtol=1e-5, kwargs...)
    local wealth_dist
    tail_width = 0.004673431

    err = Inf
    while abs(err) > rtol
    #for i=1:10
        wealth_dist = get_wealth_dist(quartile_lens, Q3_i, tail_width; kwargs...)
        mean_wealth_pred = sum(wealth_dist .* WEALTH_GRID)
        err = mean_wealth_pred/mean_wealth - 1
        tail_width += 1e-6 * err
    end
    return wealth_dist
end

"""
Compute initial wealth distribution matching empirical wealth quartiles and mean wealth.
Functional form is piecewise uniform in first three quartiles and linearly decreasing between third quartile and an upper bound.
"""
function get_wealth_grid_init(wealth_dist_df=load_empirical_wealth_dist())
    mean_wealth_init = wealth_dist_df[1, :mean_wealth_init]
    mean_wealth_overall = wealth_dist_df[1, :mean_wealth]
    Q1 = wealth_dist_df[1, :Q1] * mean_wealth_init / mean_wealth_overall
    Q2 = wealth_dist_df[1, :Q2] * mean_wealth_init / mean_wealth_overall
    Q3 = wealth_dist_df[1, :Q3] * mean_wealth_init / mean_wealth_overall

    Q1_i = findfirst(>(Q1), WEALTH_GRID_FLAT)
    Q2_i = findfirst(>(Q2), WEALTH_GRID_FLAT)
    Q3_i = findfirst(>(Q3), WEALTH_GRID_FLAT)
    len_Q1 = Q1_i - 1
    len_Q2 = Q2_i - Q1_i
    len_Q3 = Q3_i - Q2_i
    len_Q4 = N_K - Q3_i - 40
    quartile_lens = [len_Q1, len_Q2, len_Q3, len_Q4]

    wealth_dist_init = get_wealth_dist_matching_mean(quartile_lens, Q3_i, mean_wealth_init; n_empty_upper_cells=40)
    @assert abs(mean_wealth_init - sum(WEALTH_GRID .* wealth_dist_init)) < 1e-1

    return wealth_dist_init
end

function setglobal_wealth_grid_init()
    return YOUNG_WEALTH_INIT .= get_wealth_grid_init(load_empirical_wealth_dist())
end

function setglobal_z_grid_init()
    # Normal pdf sampled at state space grid points
    return YOUNG_Z_FLAT .= FLOAT.(normalize(pdf.(Normal(), LOGZ_GRID_FLAT), 1))
end

function set_globals_from_data()
    setglobal_distloc()
    setglobal_wealth_grid_init()
    setglobal_z_grid_init()
    return nothing
end
