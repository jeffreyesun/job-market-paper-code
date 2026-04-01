
using DataFrames, CSV, JLD2
import GeoDataFrames
import GeometryOps: area
using CairoMakie, GeoMakie
import Plots: scatter, savefig

include("file_paths.jl")

##############
# Clean Data #
##############

function read_cross_section(year)
    df_panel = CSV.read(PANEL_DATA_PATH, DataFrame)
    df = df_panel[df_panel.year .== year, :]
    
    @assert issorted(df[:, "2020PUMA_GISMATCH"])
    return df
end

"Impute missing housing supply elasticities based on log population density"
function impute_elasticities()
    ## Load and Merge Datasets
    gdf = GeoDataFrames.read("data/ipums_puma_2020/ipums_puma_2020.shp")
    gdf = filter(row->!(row.State in ["Alaska", "Hawaii", "Puerto Rico"]), gdf)
    gdf = DataFrame(PUMA2020=parse.(Int, gdf.GEOID), geometry=gdf.geometry)

    df_xs = read_cross_section(2020)
    df_xs = filter(row->row."2020PUMA_GISMATCH" ∈ gdf.PUMA2020, df_xs)

    df_elas = CSV.read(ELASTICITY_PATH, DataFrame)
    df_elas = DataFrame(PUMA2020=df_elas.T2020PUMA_GISMATCH, elasticity=df_elas.gamma01b_units_FMM_agg_eq20)

    @assert Set(gdf.PUMA2020) == Set(df_xs[!,"2020PUMA_GISMATCH"])
    gdf = innerjoin(gdf, df_xs, on=:"PUMA2020"=>"2020PUMA_GISMATCH")
    gdf = outerjoin(gdf, df_elas, on=:PUMA2020=>:PUMA2020)

    @assert nrow(gdf) == 2447

    ## Impute Missing Elasticities
    gdf.log_popdensity = log.(gdf.population ./ area.(gdf.geometry) .* 1e6)
    gdf.elasticity_imputed = ismissing.(gdf.elasticity)
    gdf.highdensity = gdf.log_popdensity .> 9

    gdf_reg = filter(row-> !row.highdensity & !ismissing(row.elasticity), gdf)
    intercept, slope = hcat(ones(length(gdf_reg.log_popdensity)), gdf_reg.log_popdensity)\gdf_reg.elasticity

    gdf.elasticity = [row.elasticity_imputed ? intercept + slope * row.log_popdensity : row.elasticity for row in eachrow(gdf)]

    ## Plot visualization of elasticity imputation
    f = Figure()
    a = Axis(f[1, 1]; xlabel="Log Population Density", ylabel="Housing Supply Elasticity", limits=(0,15,nothing,nothing))
    CairoMakie.scatter!(a, gdf.log_popdensity, gdf.elasticity; color=gdf.highdensity/3 + gdf.elasticity_imputed/2)
    f
    save(FIGURE_DIR*"log_popdensity_vs_elasticity_scatter.png", f)

    ## Save imputed elasticities
    CSV.write(IMPUTED_ELASTICITY_PATH, gdf[!,[:PUMA2020, :elasticity]])
    return gdf
end

"Merges data from Clean Data to Intermediate Data 1"
function merge_2020_data()
    df_elas = CSV.read(IMPUTED_ELASTICITY_PATH, DataFrame)
    df_aal = CSV.read(AAL_PATH, DataFrame)
    df_xs = read_cross_section(2020)

    df_merged_xs = innerjoin(df_elas, df_aal, on=:PUMA2020=>:puma_2020)
    df_merged_xs = innerjoin(df_merged_xs, df_xs, on="PUMA2020"=>"2020PUMA_GISMATCH")

    rename!(df_merged_xs, :housing_quantity=>:H_S, :housing_price=>:q)

    # Convert from millions USD to thousands USD, and annual to decadal.
    # Units of delta_bar are thousand USD per unit per decade.
    df_merged_xs[!,:delta_bar] = df_merged_xs.AAL_2020 ./ df_merged_xs.H_S .* 1e3 .* 10
    df_merged_xs[!,:delta_2050] = df_merged_xs.AAL_2050 ./ df_merged_xs.H_S .* 1e3 .* 10

    df_merged_xs = df_merged_xs[!,[:PUMA2020, :elasticity, :q, :population, :H_S, :homeownership_rate, :total_owner_occupied_home_value, :mean_rent, :mean_labour_earnings, :mean_labour_earnings_workingage, :mean_income, :mean_income_workingage, :delta_bar, :delta_2050]]

    CSV.write(CROSS_SECTION_PATH, df_merged_xs)
    return df_merged_xs
end

function merge_hdd_cdd_sst()
    df_hdd = CSV.read(HDD_CDD_PATH, DataFrame)
    df_sst = CSV.read(SST_PATH, DataFrame)

    periods = [string(t)*"-"*string(t+30) for t=1985:10:2065]
    midpoints = [t+15 for t=1985:10:2065]
    period_to_midpoint = Dict(periods .=> midpoints)
    transform!(df_hdd, :period=>ByRow(p->period_to_midpoint[string(p)])=>:midyear)

    df_hdd = innerjoin(df_hdd, df_sst, on=[:scenario, :midyear=>:year])

    #scatter(df_puma.SST, df_puma.HDD)
    CSV.write(MERGED_HDD_CDD_PATH, df_hdd)
    return df_hdd
end

function compute_and_separate_moments_and_params()
    df_xs = CSV.read(CROSS_SECTION_PATH, DataFrame)
    df_hdd = CSV.read(MERGED_HDD_CDD_PATH, DataFrame)
    df_sst = groupby(CSV.read(SST_PATH, DataFrame), [:year, :scenario])
    df_sensitivity = groupby(CSV.read(SENSITIVITY_PATH, DataFrame), [:parameter, :climate_driver])
    
    # Fix crosswalk to population-based crosswalk
    df_crosswalk = CSV.read(PUMA_2010_2020_CROSSWALK_PATH, DataFrame)
    df_pop20 = df_crosswalk[!, [:GEOID20, :PUMA20_Pop20]] |> unique
    df_pop20.PUMA20_Pop20 = parse.(Int, replace.(df_pop20.PUMA20_Pop20, Ref(','=>"")))
    df_xs = innerjoin(df_xs, df_pop20; on=:PUMA2020=>:GEOID20)
    df_xs.H_S .*= df_xs.PUMA20_Pop20 ./ df_xs.population
    df_xs.total_owner_occupied_home_value .*= df_xs.PUMA20_Pop20 ./ df_xs.population
    df_xs.population .= df_xs.PUMA20_Pop20

    SST_RCP45_2020 = df_sst[(2020, "RCP45")].SST |> only
    SST_RCP45_2050 = df_sst[(2050, "RCP45")].SST |> only
    ΔSST_2020_2050 = SST_RCP45_2050 - SST_RCP45_2020

    get_slope(x, y) = (hcat(ones(length(x)), x)\y)[2:2]

    gdf_hdd = groupby(df_hdd, :GEOID)
    df_hdd_g = combine(gdf_hdd, [:SST,:HDD]=>get_slope=>[:HDD_g], [:SST,:CDD]=>get_slope=>[:CDD_g])

    df_xs = innerjoin(df_xs, df_hdd_g, on=:PUMA2020=>:GEOID)

    df_xs.delta_g = (log.(df_xs.delta_2050) - log.(df_xs.delta_bar))/ΔSST_2020_2050
    df_xs.A_g = df_xs.HDD_g * df_sensitivity[("A", "HDD")].sensitivity[1] .+ df_xs.CDD_g * df_sensitivity[("A", "CDD")].sensitivity[1]
    df_xs.α_g = df_xs.HDD_g * df_sensitivity[("alpha", "HDD")].sensitivity[1] .+ df_xs.CDD_g * df_sensitivity[("alpha", "CDD")].sensitivity[1]

    df_params = df_xs[!,[:PUMA2020, :elasticity, :delta_bar, :A_g, :α_g, :delta_g, :HDD_g, :CDD_g]]
    df_moments_2020 = df_xs[!,[:PUMA2020, :q, :population, :H_S, :homeownership_rate, :total_owner_occupied_home_value, :mean_rent, :mean_labour_earnings, :mean_labour_earnings_workingage, :mean_income, :mean_income_workingage]]
    df_moments_2020[!,:delta] = df_xs.delta_bar

    CSV.write(SPATIAL_PARAMS_PATH, df_params)
    CSV.write(SPATIAL_MOMENTS_2020_PATH, df_moments_2020)
    return (;df_params, df_moments_2020)
end

function subset_and_save_pairwise_distances()
    df_dist = CSV.read(DISTANCE_PATH, DataFrame)
            
    pumas = CSV.read(SPATIAL_PARAMS_PATH, DataFrame).PUMA2020

    @assert parse.(Int, names(df_dist)[2:end]) == df_dist[:,1]
    @assert issorted(df_dist[:,1])
    @assert all(in(df_dist[:,1]), pumas)

    mask = in.(df_dist[:,1], Ref(pumas))
    mat_dist = Matrix(df_dist[:, 2:end])
    mat_dist = mat_dist[mask, mask]

    save(CLEANED_DISTANCE_PATH, "pairwise_distances", mat_dist)
    return mat_dist
end


#################################
# Generate Some Plots for Paper #
#################################

function plot_hdd_cdd_vs_sst()
    df_hdd = CSV.read(MERGED_HDD_CDD_PATH, DataFrame)
    pumas = [4500800, 3402202, 3604308, 102802, 1700100, 603709, 3901202, 3700800, 5116500, 1800800]
    
    for puma in pumas
        df_puma = df_hdd[df_hdd.GEOID .== puma, :]
        fig_hdd = scatter(df_puma.SST, df_puma.HDD; xlabel="SST", ylabel="HDD", legend=false, title="PUMA "*string(puma))
        savefig(fig_hdd, FIGURE_DIR*"hdd_vs_sst_puma_"*string(puma)*".png")
        fig_cdd = scatter(df_puma.SST, df_puma.CDD; xlabel="SST", ylabel="CDD", legend=false, title="PUMA "*string(puma))
        savefig(fig_cdd, FIGURE_DIR*"cdd_vs_sst_puma_"*string(puma)*".png")
    end    
end




#######
# API #
#######

function clean_data()
    impute_elasticities()
    merge_2020_data()
    merge_hdd_cdd_sst()
    compute_and_separate_moments_and_params()
    subset_and_save_pairwise_distances()
    return nothing
end

function generate_plots()
    plot_hdd_cdd_vs_sst()
end

function prepare_data()
    clean_data()
    generate_plots()
end
