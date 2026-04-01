
"""
Run all analyses in the paper.

Before running from freshly cloned repo, unpack `data/1 Cleaned Data/puma_2020_pairwise_geodesic_distances.csv.7z`.

## Some places where things are set (within `code/within-period model`):
- Constants controlling floating-point precision, GPU usage, and grid sizes are all in `main.jl`.
- Parameter defaults are in `types.jl`.
- Some additional parameters are in `params.jl`.
- Many globals are set in `initialize_globals.jl`
- Many large data structures are defined in `preallocate.jl`
- Many GPU-specific types and constants are set in `types_gpu.jl`
"""

using Dates: now
include("code/main.jl")
include("clean_data.jl")


###########
# Options #
###########

T = 20 # Transition length
PREPARE_DATA = true
LOAD_INTERMEDIATE_CALIBRATION = false
LOAD_COMPLETED_CALIBRATION = false
LOAD_TRANSITION_PRICES = false
LOAD_INSURANCE_TRANSITION_PRICES = false
LOAD_Π_POLICY_TRANSITION_PRICES = false
LOAD_MIGRATION_TRANSITION_PRICES = false
LOAD_PERFECT_FORESIGHT_VNET = false


##########################################
# Free Up Memory Between Counterfactuals #
##########################################

"Delete and garbage collect the large data structures, especially `td`, the transition data."
function free_ram()
    global td = nothing
    global pp = nothing
    global pd = nothing
    sleep(2)
    GC.gc()
    sleep(2)
end


################
# Prepare Data #
################

if PREPARE_DATA
    clean_data()
    generate_plots()
end


#############
# Calibrate #
#############

println("Calibrating model...")
# Calibrate from scratch
if LOAD_COMPLETED_CALIBRATION
    println("Loading completed calibration...")
    (;pd, params) = load_fully_calibrated_params()
else
    params = Params(read_spatial_params_initial_guess())
    pd = PeriodData(read_2020_prices(), params)
    LOAD_INTERMEDIATE_CALIBRATION && load_temp_params!(pd, params)
end
skip_params = [:ψ, :σ]
N_LOC < 100 && push!(skip_params, :F_u_dist)
calibrate_model!(pd, params; verbosity=2, skip_params, save=true, save_intermediate=true, update_speed=0.01)


###############################
# Solve Baseline Steady State #
###############################

println("Solving baseline steady state...")
(;pd, params) = load_fully_calibrated_params()
solve_steady_state!(pd, params; verbosity=2, update_speed=10, rtol=1e-3)


######################################
# Solve Perfect Foresight Transition #
######################################

println("Solving perfect foresight transition...")
(;pd, params) = load_fully_calibrated_params()
pp = get_climate_change_counterfactual_params(params, T)

if LOAD_TRANSITION_PRICES
    (;q, q_last, ρ) = read_transition_prices(pp)
    (;td, pp) = solve_transition(pp; q, q_last, ρ, verbosity=2)
else
    solve_steady_state!(pd, params; verbosity=1, update_speed=10)
    (;td, pp) = solve_transition(pp; verbosity=2)
end

write_transition_moments(td, pp; save=true)

free_ram()


##########################################
# Compute Flood Insurance Counterfactual #
##########################################

println("Solving flood insurance counterfactual transition...")
(;pd, params) = load_fully_calibrated_params()
pp = get_climate_change_counterfactual_params(params, T; δ_factor=1.07)

if LOAD_INSURANCE_TRANSITION_PRICES
    (;q, q_last, ρ) = read_transition_prices(pp; suffix="_flood_insurance")
    (;td, pp) = solve_transition(pp; q, q_last, ρ, verbosity=2)
else
    (;q, q_last, ρ) = read_transition_prices(pp)
    solve_steady_state!(pd, params; verbosity=1, update_speed=10)
    (;td, pp) = solve_transition(pp; q, q_last, ρ, verbosity=2)
end

write_transition_moments(td, pp; save=true, suffix="_flood_insurance")

free_ram()


#########################################
# Compute Housing Supply Counterfactual #
#########################################

println("Solving housing supply counterfactual transition...")
(;pd, params) = load_fully_calibrated_params()
pp = get_climate_change_Π_policy_counterfactual_params(params, T; treated_Π_factor=1.05)

if LOAD_Π_POLICY_TRANSITION_PRICES
    (;q, q_last, ρ) = read_transition_prices(pp; suffix="_housing_supply")
    (;td, pp) = solve_transition(pp; q, q_last, ρ, verbosity=2)
else
    solve_steady_state!(pd, params; verbosity=1, update_speed=10)
    (;td, pp) = solve_transition(pp; verbosity=2)
end

write_transition_moments(td, pp; save=true, suffix="_housing_supply")

free_ram()


#########################################
# Compute Migration Ease Counterfactual #
#########################################

println("Solving migration ease counterfactual transition...")
(;pd, params) = load_fully_calibrated_params()
params = get_migration_ease_counterfactual_params(params)
pp = [deepcopy(params) for t=1:T]

if LOAD_MIGRATION_TRANSITION_PRICES
    (;q, q_last, ρ) = read_transition_prices(pp; suffix="_migration_ease")
    (;td, pp) = solve_transition(pp; q, q_last, ρ, verbosity=2)
else
    solve_steady_state!(pd, params; verbosity=1, update_speed=10)
    (;td, pp) = solve_transition(pp; verbosity=2)
end

write_transition_moments(td, pp; save=true, suffix="_migration_ease")

free_ram()


########################################################
# Train Neural Network on Perfect Foresight Transition #
########################################################

println("Pre-training on deterministic transition...")

# Load solved transition
(;pd, params) = load_fully_calibrated_params()
pp = get_climate_change_counterfactual_params(params, 20)
(;q, q_last, ρ) = read_transition_prices(pp)
(;td, pp) = solve_transition(pp; q, q_last, ρ)

# Initialize vnet and training data
vnet = gpu(VNet(;width=512))
LOAD_PERFECT_FORESIGHT_VNET && load_vnet!(vnet; suffix="_perfect_foresight")
train_set = get_training_data(td, pp; stochastic=false)

# Train vnet
train_vnet!(vnet, ()->train_set; epochs=400, verbosity=2, lr=1e-4, batch_size=10, save_suffix="_perfect_foresight_tmp")
save_vnet(vnet; suffix="_perfect_foresight")

# Compare V_end predictions to conventionally-solved V_end
vnet(gpu(train_set[1][1]))[:,:,:,:,5]
train_set[1][2][:,:,:,:,5]

# Compute Perfect Foresight Transition Using Neural Net #
#-------------------------------------------------------#
# Compute full perfect foresight transition using vnet
simulate_path_given_vnet!(td, pp, vnet; SST_process=:deterministic, rtol=1e-4, verbosity=2)
write_transition_moments(td, pp; save=true, suffix="_perfect_foresight_vnet")

free_ram()


#############################################
# Train Neural Net on Stochastic Transition #
#############################################

println("Training on stochastic transition...")

# Load solved transition
(;pd, params) = load_fully_calibrated_params()
pp = get_climate_change_counterfactual_params(params, 20)
(;q, q_last, ρ) = read_transition_prices(pp)
(;td, pp) = solve_transition(pp; q, q_last, ρ)

# Load vnet trained on perfect foresight transition
vnet = gpu(VNet(;width=512))
#load_vnet!(vnet; suffix="_perfect_foresight")
load_vnet!(vnet; suffix="_tmp")

# Define training data
function get_train_set_stoch(; kwargs...)
    open(io->println(io, now(), " Started get_train_set_stoch"), "train_set_stoch.log", "a")
    train_set = get_training_data(td, pp; vnet, stochastic=true, verbosity=2, rtol=1e-3, update_speed=50, kwargs...)
    open(io->println(io, now(), " Finished get_train_set_stoch"), "train_set_stoch.log", "a")
    return train_set
end

tot_loss = Inf
while tot_loss > 1e-5
    (;tot_loss, vnet) = train_vnet!(vnet, get_train_set_stoch; epochs=100, verbosity=2, lr=1e-5, batch_size=10)
    save_vnet(vnet; suffix="_eon_tmp")
end
save_vnet(vnet; suffix="_uncertainty")


##################################################
# Compute Median Path Uncertainty Counterfactual #
##################################################

# Generate moments for global solution on median climate path
for params_t=pp
    params_t.scenario_ind = SCENARIO_IND_INIT
end
td = simulate_path_given_vnet!(td, pp, vnet; SST_process=:deterministic) 
write_transition_moments(td, pp; save=true, suffix="_uncertainty")



######################################
# Compute News Shock Counterfactuals #
######################################

# Generate moments for global solution on median climate path, with one bad surprise
for (t,params_t) in enumerate(pp)
    params_t.scenario_ind = t <= 3 ? SCENARIO_IND_INIT : N_SCENARIO
end
td = simulate_path_given_vnet!(td, pp, vnet; SST_process=:deterministic) 
write_transition_moments(td, pp; save=true, suffix="_bad_news_shock")

# Generate moments for global solution on median climate path, with one good surprise
for (t,params_t) in enumerate(pp)
    params_t.scenario_ind = t <= 3 ? SCENARIO_IND_INIT : 1
end
td = simulate_path_given_vnet!(td, pp, vnet; SST_process=:deterministic) 
write_transition_moments(td, pp; save=true, suffix="_good_news_shock")
