
using DataFrames
using CSV
import Distances

###########
# Climate #
###########

"Merge in empirical climate sensitivities from data."
function add_climate_sensitivities!(loc_grid::LocGrid, df_climate::DataFrame)
    vec(loc_grid.elasticity) .= df_climate[!,:elasticity]
    vec(loc_grid.A_g) .= .- df_climate[!, :prod_logdiff_perC]
    α_pctdam = df_climate[!, :heat_disamenity_pctincome_perC] + df_climate[!, :cold_disamenity_pctincome_perC]
    vec(loc_grid.α_g) .= @. log(1 - α_pctdam/100)
    vec(loc_grid.δ_g) .= @. log(1 + df_climate[!, :AAL_pctinc_perC]/100)

    return loc_grid
end

###############
# Calibration #
###############

# Empirical Equilibrium #
#-----------------------#
const EMPIRICAL_DIRPATH = "data/empirical_steady_state/"
"Generate the filepath for the empirical equilibrium."
function empirical_steady_state_filepath(locunit::String, year::Int)
    fn = join([locunit, year], '_')
    return EMPIRICAL_DIRPATH*fn*".csv"
end

"Read data from empirical equilibrium."
function read_empirical_steady_state_df(n_loc=N_LOC; locunit="1990PUMA", year=1990)
    fp = empirical_steady_state_filepath(locunit, year)
    df = CSV.File(fp) |> DataFrame
    df = df[.!ismissing.(df[!,:ZHVI]),:]
    df = df[.!ismissing.(df[!,:prod_logdiff_perC]), :]
    sort!(df, :pop_1990; rev=true)
    return df[1:n_loc,:]
end

"Generate loc_grid from empirical equilibrium."
function read_empirical_steady_state(df::DataFrame)
    n_loc = N_LOC
    GISMATCH = df[!, :GISMATCH]

    # Fundamentals
    A = fill(NaN, n_loc)
    α = fill(NaN, n_loc)
    δ = @. df[!,:AAL_2020]*1e6/df[!,:H_qual]*10 / 1000

    # Prices
    q_last = q = df[!,:q] / 1000
    ρ = fill(NaN, n_loc)

    # Quantities
    pop = df[!,:pop_1990]
    scale = n_loc ./ sum(pop)
    pop .*= scale
    H = df[!,:H_qual] .* scale

    loc_grid_flat = StructArray(Location.(
        GISMATCH, A, α, δ, q, ρ, pop, H, q_last
    ))
    loc_grid = pad_dims(loc_grid_flat; left=4, ndims_new=5)
    return add_climate_sensitivities!(loc_grid, df)
end
function read_empirical_steady_state(n_loc::Int=N_LOC; locunit="1990PUMA", year=1990)
    read_empirical_steady_state(read_empirical_steady_state_df(n_loc; locunit, year))
end

"Generate loc_target_moments from empirical equilibrium."
function read_empirical_target_moments(df::DataFrame)
    GISMATCH = df[!, :GISMATCH]

    homeown_frac = df[!, :homeown_frac]
    total_ownocc_valueh = df[!, :total_owned_valueh]*1e6
    mean_rent = df[!, :mean_rent]*120
    mean_earn = df[!, :mean_earn]*10
    mean_earn_workage = df[!, :mean_earn_workage]*10
    median_rent_to_earn = df[!, :median_rent_to_earn]

    loc_target_moments = StructArray{LocationMoments{FLOAT_PRECISION}}((
        GISMATCH,
        Vector{FLOAT_PRECISION}.([
            homeown_frac, total_ownocc_valueh, mean_rent,
            mean_earn, mean_earn_workage, median_rent_to_earn
        ])...
    ))
    loc_target_moments_grid = pad_dims(loc_target_moments; left=4, ndims_new=5)

    agg_moments_df = CSV.File("data/empirical_steady_state/aggregate_moments.csv") |> DataFrame
    agg_moments = Dict(String.(agg_moments_df.moment) .=> agg_moments_df.value)
    agg_moments["median_rent_to_earn"] = mean(loc_target_moments.median_rent_to_earn)
    agg_moments["share_H_sold"] = 0.6513
    for k in ["move_rate_PUMA", "move_rate_state", "move_rate_young_PUMA", "move_rate_young_state"]
        agg_moments[k] = 1 - (1-agg_moments[k])^2
    end
    agg_moments = Dict(k => FLOAT_PRECISION(agg_moments[k]) for k in keys(agg_moments))
    return loc_target_moments_grid, agg_moments
end
function read_empirical_target_moments(n_loc::Int=N_LOC; locunit="1990PUMA", year=1990)
    read_empirical_target_moments(read_empirical_steady_state_df(n_loc; locunit, year))
end

function get_empirical_aggregate_moments()
    loc_grid = read_empirical_steady_state()
    loc_target_moments, agg_moments = read_empirical_target_moments()

    return AggMoments(
        rent_share_earn = mean(loc_target_moments.median_rent_to_earn),
        homeown_rate = sum(loc_target_moments.homeown_frac .* loc_grid.pop) / sum(loc_grid.pop),
        share_H_sold = agg_moments["share_H_sold"],
        share_pop_moving = agg_moments["move_rate_PUMA"],
        share_young_moving = agg_moments["move_rate_young_PUMA"],
        mean_wealth_homeown_over_mean = agg_moments["mean_wealth_homeown"] / agg_moments["mean_wealth"],
        mean_wealth_oldest_over_mean = agg_moments["mean_wealth_old"] / agg_moments["mean_wealth"],
    )
end

"Import pairwise distance data."
function read_pairwise_distances(df::DataFrame=read_empirical_steady_state_df())
    lat = df[!, :lat] ./ 1e3
    long = df[!, :long] ./ 1e3
    area = df[!, :aland] .* 1e6
    area *= 8080464.3 / sum(area)

    centroids = collect(zip(Float64.(lat), Float64.(long)))
    dists = Distances.pairwise(Distances.Haversine(), centroids)

    dists .+= Diagonal(@. sqrt(area) * 128/(45π))
    return dists
end

###################################
# Location Names and Constructors #
###################################

# Names #
#-------#
function read_1990PUMA_names()
    PUMA_names = CSV.File("data/1990_PUMAs_5pct.csv") |> DataFrame
    #PUMA_names = unique(PUMA_names[!, ["State", "PUMA", "County name"]])
    PUMA_names = PUMA_names[!, ["PUMA", "State", "County name"]]

    state_names = CSV.File("data/stateFIPS.csv") |> DataFrame
    state_names = Dict(state_names[!, :stateFIPS] .=> state_names[!, "state name"])

    PUMA_names[!, "State name"] = get.(Ref(state_names), PUMA_names[!, :State], "")
    PUMA_names[!, "State name"] = uppercasefirst.(lowercase.(PUMA_names[!, "State name"]))
    PUMA_names[!, :GISMATCH] = @. PUMA_names[!, :State]*100000 + PUMA_names[!, :PUMA]

    PUMA_names = PUMA_names[!, ["GISMATCH", "County name", "State name"]]
    return rename!(PUMA_names, ["GISMATCH", "county", "state"])
end

const _1990PUMA_NAMES = read_1990PUMA_names()

function PUMA_name(GISMATCH, PUMA_names=_1990PUMA_NAMES)
    name = PUMA_names[PUMA_names[!, :GISMATCH] .== GISMATCH, :]
    return "$(name[1,:county]), $(name[1,:state])"
end

PUMA_name(loc::Location, PUMA_names=_1990PUMA_NAMES) = PUMA_name(loc.GISMATCH, PUMA_names)
#PUMA_name_i(i::Int, loc_grid_empirical) = PUMA_name(loc_grid_empirical[i].GISMATCH)


#################################
# Empirical Wealth Distribution #
#################################

function get_wealth_grid_init(agg_moments)
    #wealth_path = get_mean_wealth_path(pd)
    #young_factor = wealth_path[1] / mean(wealth_path) / 1000
    young_factor = 0.0007313892f0
        
    target_mean_wealth_init = agg_moments["mean_wealth"] * young_factor
    Q1 = agg_moments["bottom_quartile"] * young_factor
    Q2 = agg_moments["median"] * young_factor
    Q3 = agg_moments["top_quartile"] * young_factor

    Q1_i = findfirst(>(Q1), WEALTH_GRID_FLAT)
    Q2_i = findfirst(>(Q2), WEALTH_GRID_FLAT)
    Q3_i = findfirst(>(Q3), WEALTH_GRID_FLAT)
    len_Q1 = Q1_i - 1
    len_Q2 = Q2_i - Q1_i
    len_Q3 = Q3_i - Q2_i
    len_Q4 = N_K - Q3_i
    quartile_lens = [len_Q1, len_Q2, len_Q3, len_Q4]

    wealth_init = Float32.(vcat([fill(1/len_Q/4, len_Q) for len_Q in quartile_lens]..., 0))
    tilter = range(-1, 1, length=N_K-Q3_i)
    @views wealth_init[Q3_i:N_K-1] .-= 0.001963 .* tilter
    mean_wealth_init = sum(WEALTH_GRID .* wealth_init)
    @assert abs(mean_wealth_init - target_mean_wealth_init) < 1e-1

    return wealth_init ./ N_AGE
end


#########################
# Climate Sensitivities #
#########################

"""
Set location fundamentals
`Π, H_bar, A_bar, α_bar, δ_bar`
to the current values.
"""
function set_fundamentals_as_baseline_year!(loc_grid::LocGrid)
    (;A, α, δ, q, H_D) = loc_grid
    # Climate
    loc_grid.A_bar .= A
    loc_grid.α_bar .= α
    loc_grid.δ_bar .= δ
    # Housing
    loc_grid.Π .= q
    loc_grid.H .= H_D
    loc_grid.H_bar .= H_D
    return loc_grid
end


##########################
# Steady State Solutions #
##########################

# File Paths #
#------------#
const SS_SOLN_DIRPATH = "data/steady_state_solutions/"
const TRANS_SOLN_DIRPATH = "data/midtransition_solutions/"

"Generate the filepath for the steady state solution."
function steady_state_solution_filepath(locunit::String, n_loc::Int, sim_year::Int, RCP::String)
    strfields = [locunit, string(n_loc)*"loc", sim_year]
    RCP != "" && push!(strfields, RCP)
    fn = join(strfields, '_')
    return SS_SOLN_DIRPATH*fn*".csv"
end

function midtransition_solution_filepath(locunit, n_loc, start_year, end_year, sim_year, RCP)
    fn = "$(locunit)_$(n_loc)loc_$(start_year)-$(end_year)_at_$(sim_year)_$(RCP)"
    return TRANS_SOLN_DIRPATH*fn*".csv"
end

function branched_ss_filepath(RCP_new)
    fn = "$(RCP_new)_$(N_LOC)loc"
    return "data/branched_steady_states/"*fn*".csv"
end

function transition_panel_filepath(RCP)
    fn = "$(RCP)_$(N_LOC)loc"
    return "data/transition_panels/"*fn*".csv"
end

function surprise_transition_filepath(RCP_original, RCP_new)
    fn = "$(RCP_original)_to_$(RCP_new)_$(N_LOC)loc"
    return "data/surprise_transition_panels/"*fn*".csv"
end

# Write Solution #
#----------------#
function DataFrames.DataFrame(loc_grid::LocGrid, year=nothing)
    write_cols = collect(fieldnames(Location))
    df = (write_cols .=> vec.(getproperty.(Ref(loc_grid), write_cols))) |> DataFrame
    isnothing(year) && return df
    df[!, :year] .= year
    return select!(df, :year, Not([:year]))
end

function DataFrames.DataFrame(loc_grid_path::LocGridPath, start_year, end_year)
    dfs = DataFrame.(get_loc_grid_vec(loc_grid_path), start_year:10:end_year)
    return vcat(dfs...)
end

"Save steady state solution to file."
function write_steady_state_solution(loc_grid::LocGrid, sim_year::Int, RCP::String; locunit="1990PUMA")
    n_loc = size(loc_grid, LOC_DIM)
    fn = steady_state_solution_filepath(locunit, n_loc, sim_year, RCP)
    return CSV.write(fn, DataFrame(loc_grid))
end

function write_steady_state_solution_baseline_year(loc_grid::LocGrid, sim_year::Int, RCP::String)
    set_fundamentals_as_baseline_year!(loc_grid)
    return write_steady_state_solution(loc_grid, sim_year, RCP)
end

function write_midtransition_solution(loc_grid::LocGrid, start_year, end_year, sim_year, RCP::String; locunit="1990PUMA")
    n_loc = size(loc_grid, LOC_DIM)
    fn = midtransition_solution_filepath(locunit, n_loc, start_year, end_year, sim_year, RCP)
    return CSV.write(fn, DataFrame(loc_grid))
end

function write_transition_panel(loc_grid_path::LocGridPath, start_year, end_year, RCP::String)
    fn = transition_panel_filepath(RCP)
    return CSV.write(fn, DataFrame(loc_grid_path, start_year, end_year))
end

function write_surprise_transition_panel(loc_grid_path, start_year, end_year, sim_year, RCP_original, RCP_new)
    fn = surprise_transition_filepath(RCP_original, RCP_new)
    return CSV.write(fn, DataFrame(loc_grid_path, sim_year, end_year))
end

# Read Solution #
#---------------#
function LocGrid(df::DataFrame, n_loc::Int=N_LOC)
    loc_grid = LocGrid(n_loc)
    for c=fieldnames(Location)
        vec(getproperty(loc_grid, c)) .= df[!, c]
    end
    #loc_grid.q_last .= loc_grid.q
    @assert !any(isnan, loc_grid.q_last)
    return loc_grid
end

"Read steady state solution from file."
function read_steady_state_solution_df(sim_year::Int, RCP::String; locunit="1990PUMA", n_loc=N_LOC)
    fn = steady_state_solution_filepath(locunit, n_loc, sim_year, RCP)
    df_est = DataFrame(CSV.File(fn))
    return df_est[1:N_LOC,:]
end

function read_steady_state_solution_locgrid(sim_year::Int, RCP::String, n_loc::Int=N_LOC; locunit="1990PUMA")
    df_ss = read_steady_state_solution_df(sim_year, RCP; locunit, n_loc)
    return LocGrid(df_ss, n_loc)
end

function read_branched_ss_solution(RCP_new)
    fn = branched_ss_filepath(RCP_new)
    df = DataFrame(CSV.File(fn))[!, :]
    return LocGrid(df)
end

function write_branched_ss_solution(loc_grid::LocGrid, RCP_new)
    n_loc = size(loc_grid, LOC_DIM)
    fn = branched_ss_filepath(RCP_new)
    write_cols = collect(fieldnames(Location))
    df_est = (write_cols .=> vec.(getproperty.(Ref(loc_grid), write_cols))) |> DataFrame
    df_est[!,:q_last] .= df_est[!,:q]
    return CSV.write(fn, df_est)
end

function read_midtransition_solution(start_year, end_year, sim_year::Int, RCP::String, n_loc=N_LOC; locunit="1990PUMA")
    fn = midtransition_solution_filepath(locunit, n_loc, start_year, end_year, sim_year, RCP)
    df = DataFrame(CSV.File(fn))[1:n_loc, :]
    return LocGrid(df, n_loc)
end

function LocGridPath(df::DataFrame)
    start_year = minimum(df[!,:year])
    end_year = maximum(df[!,:year])
    dfs = [df[df[!,:year].==year, :] for year in start_year:10:end_year]
    return LocGridPath(LocGrid.(dfs))
end
LocGridPath(csv::CSV.File) = LocGridPath(DataFrame(csv))

function read_transition_panel(start_year, end_year, RCP)
    fn = transition_panel_filepath(RCP)
    df = CSV.File(fn) |> DataFrame
    return LocGridPath(df)
end

function read_surprise_transition_panel(branch_year, end_year, RCP_original, RCP_new)
    fn = surprise_transition_filepath(RCP_original, RCP_new)
    df = CSV.File(fn) |> DataFrame
    dfs = [df[df[!,:year].==year, :] for year in branch_year:10:end_year]
    return LocGridPath(LocGrid.(dfs))
end
