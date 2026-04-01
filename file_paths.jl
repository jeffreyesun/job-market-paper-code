
# Cleaned Raw Data
global CLEANED_DATA_DIR = "data/1 Cleaned Data/"
global PANEL_DATA_PATH = CLEANED_DATA_DIR*"panel_data.csv"
global DISTANCE_PATH = CLEANED_DATA_DIR*"puma_2020_pairwise_geodesic_distances.csv"
global ELASTICITY_PATH = CLEANED_DATA_DIR*"Supply_elasticities.csv"
global WEALTH_DIST_PATH = CLEANED_DATA_DIR*"wealth_dist.csv"
global AGGREGATE_MOMENTS_PATH = CLEANED_DATA_DIR*"aggregate_moments.csv"
global PUMA_2010_2020_CROSSWALK_PATH = CLEANED_DATA_DIR*"PUMA2010_PUMA2020_crosswalk.csv"

# Climate Data
global CLIMATE_DIR = "data/climate/"
global HDD_CDD_PATH = CLIMATE_DIR*"hdd_cdd_puma_panel_weighted.csv"
global AAL_PATH = CLIMATE_DIR*"AAL.csv"
global SST_PATH = CLIMATE_DIR*"SST.csv"
global SENSITIVITY_PATH = CLIMATE_DIR*"climate_sensitivities.csv"

# Intermediate Data
global INTERMEDIATE_DATA_1_DIR = "data/2 Intermediate Data/"
global IMPUTED_ELASTICITY_PATH = INTERMEDIATE_DATA_1_DIR*"imputed_elasticities.csv"
global CROSS_SECTION_PATH = INTERMEDIATE_DATA_1_DIR*"cross_section_2020.csv"
global MERGED_HDD_CDD_PATH = INTERMEDIATE_DATA_1_DIR*"hdd_cdd_merged.csv"

# Moments and Partial Params
global INTERMEDIATE_DATA_2_DIR = "data/3 Moments and Partial Params/"
global SPATIAL_PARAMS_PATH = INTERMEDIATE_DATA_2_DIR*"spatial_params.csv"
global SPATIAL_MOMENTS_2020_PATH = INTERMEDIATE_DATA_2_DIR*"spatial_moments_2020.csv"
global CLEANED_DISTANCE_PATH = CLEANED_DATA_DIR*"puma_2020_pairwise_geodesic_distances.jld2"

# Spatially but Not Fully Calibrated Prices and Params
global INTERMEDIATE_DATA_3_DIR = "data/4 Spatially Calibrated Params/"

# Fully Calibrated Prices and Params
global INTERMEDIATE_DATA_4_DIR = "data/5 Fully Calibrated Params/"

# Neural Network Parameters
global NEURAL_NET_PARAMS_DIR = "data/6 Neural Net Params/"

# Figures
global FIGURE_DIR = "figures/"

# Model Output
global SS_SOLN_DIRPATH = "data/model_output/"
global TRANSITION_MOMENTS_DIRPATH = "output/transition_moments/"

# Panel Simulations
global TRANSITION_PANEL = TRANSITION_MOMENTS_DIRPATH*"transition_panel_2447loc.csv"
global TRANSITION_PANEL_FLOODINSURANCE = TRANSITION_MOMENTS_DIRPATH*"transition_panel_2447loc_flood_insurance.csv"
global TRANSITION_PANEL_HOUSINGSUPPLY = TRANSITION_MOMENTS_DIRPATH*"transition_panel_2447loc_housing_supply.csv"
global TRANSITION_PANEL_PERFECTFORESIGHT_VNET = TRANSITION_MOMENTS_DIRPATH*"transition_panel_2447loc_perfect_foresight_vnet.csv"
global TRANSITION_PANEL_UNCERTAINTY = TRANSITION_MOMENTS_DIRPATH*"transition_panel_2447loc_uncertainty.csv"
global TRANSITION_PANEL_BADNEWS = TRANSITION_MOMENTS_DIRPATH*"transition_panel_2447loc_bad_news_shock.csv"
global TRANSITION_PANEL_GOODNEWS = TRANSITION_MOMENTS_DIRPATH*"transition_panel_2447loc_good_news_shock.csv"


# Aggregate Time Series Simulations
global TRANSITION_SERIES = TRANSITION_MOMENTS_DIRPATH*"transition_series_2447loc.csv"
global TRANSITION_SERIES_FLOODINSURANCE = TRANSITION_MOMENTS_DIRPATH*"transition_series_2447loc_flood_insurance.csv"
global TRANSITION_SERIES_HOUSINGSUPPLY = TRANSITION_MOMENTS_DIRPATH*"transition_series_2447loc_housing_supply.csv"
global TRANSITION_SERIES_PERFECTFORESIGHT_VNET = TRANSITION_MOMENTS_DIRPATH*"transition_series_2447loc_perfect_foresight_vnet.csv"
global TRANSITION_SERIES_UNCERTAINTY = TRANSITION_MOMENTS_DIRPATH*"transition_series_2447loc_uncertainty.csv"
global TRANSITION_SERIES_BADNEWS = TRANSITION_MOMENTS_DIRPATH*"transition_series_2447loc_bad_news_shock.csv"
global TRANSITION_SERIES_GOODNEWS = TRANSITION_MOMENTS_DIRPATH*"transition_series_2447loc_good_news_shock.csv"
