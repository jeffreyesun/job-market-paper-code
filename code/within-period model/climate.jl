
"""
Define a global climate process which acts as an AR(1) process for global temperature,
but with different grids for each decade.
"""
module ClimateProcess
export N_SCENARIO, SCENARIO_IND_INIT, SST_TRANSMAT, SST_GRID, draw_scenario_ind, draw_scenario_path

const N_SCENARIO = 5
const SCENARIO_IND_INIT = 3
const PERSISTENCE = 0.95

using CSV, DataFrames
using QuantEcon: rouwenhorst 
using Distributions: Categorical, Normal, quantile, cdf

# Import Data #
sst_df = CSV.File("data/climate/WPD_Lyon.csv") |> DataFrame

# Convert to deviations from 2020
sst_df = sst_df[!, Not(:RCP26)]
for col in [:RCP45, :RCP60]
    sst_df[!, col] .-= sst_df[1, col]
end

# Interpolate decades 2100-2500
for year=2100:10:2500
    if year%50 != 0
        l = year - year%50
        r = l + 50
        w = 1 - year%50/50
        lrow = only(eachrow(sst_df[sst_df.year .== l, :]))
        rrow = only(eachrow(sst_df[sst_df.year .== r, :]))
        RCP45 = w*lrow.RCP45 + (1-w)*rrow.RCP45
        RCP60 = w*lrow.RCP60 + (1-w)*rrow.RCP60
        push!(sst_df, (;year, RCP45, RCP60))
    end
end
sort!(sst_df, :year)

# Define an AR(1) process with standard normal innovations #
ar1_process = rouwenhorst(N_SCENARIO, PERSISTENCE, 1, 0)
ar1_grid = ar1_process.state_values
ar1_transmat = ar1_process.p

# Compute means and stds of normal distribution with RCP45 and RCP60 as 1/3 and 2/3 quantiles #
sst_df[:, :mean] = (sst_df.RCP45 + sst_df.RCP60) / 2
sst_df[:, :std] = (sst_df.RCP60 - sst_df.RCP45) / (quantile(Normal(), 2/3) - quantile(Normal(), 1/3))

for row in eachrow(sst_df)[2:end]
    @assert cdf(Normal(row.mean, row.std), row.RCP45) ≈ 1/3
    @assert cdf(Normal(row.mean, row.std), row.RCP60) ≈ 2/3
end

# Convert to matrix #
sst_dist = sst_df[:, [:mean, :std]]
@assert sst_dist[1,1] == 0
big_T = size(sst_dist, 1)

# Compute the conditional variances of the AR(1) RV given the initial state #
ar1_dist_path = [setindex!(zeros(N_SCENARIO), 1, SCENARIO_IND_INIT)]
for t=1:big_T-1
    push!(ar1_dist_path, ar1_transmat'ar1_dist_path[end])
end
ar1_dist_path = reduce(hcat, ar1_dist_path)
ar1_std_path = sqrt.([(ar1_grid.^2)'dist for dist in eachcol(ar1_dist_path)])

# Convert the conditional AR(1) RV into a process for SST #
function get_SST(t, ar_val)
    t == 1 && return sst_dist[1,1]
    projection_deviation = ar_val/ar1_std_path[t]
    return SST = sst_dist[t,1] + sst_dist[t,2] * projection_deviation
end

const SST_GRID = Float32.(get_SST.((1:big_T)', ar1_grid))
const SST_TRANSMAT = Float32.(ar1_transmat)

# Check that, with SST_GRID, the AR(1) process has the moments sst_df.mean and sst_df.std #
@assert sum(ar1_dist_path.*SST_GRID; dims=1)' ≈ sst_df.mean
std_pred = sqrt.(sum(ar1_dist_path.*(SST_GRID.-sst_df.mean').^2; dims=1))'
@assert maximum(abs.(std_pred .- sst_df.std)) < 1e-7

"Draw the next climate scenario index given the last one."
draw_scenario_ind(scenario_ind_last) = rand(Categorical(SST_TRANSMAT[scenario_ind_last, :]))
"Draw a full scenario path of length T given an initial scenario index."
function draw_scenario_path(scenario_ind_init, T)
    SST_path = [scenario_ind_init]
    for t=1:T-1
        push!(SST_path, draw_scenario_ind(SST_path[end]))
    end
    return SST_path
end
end
