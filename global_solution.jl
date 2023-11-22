using Flux
using ChainRules
import Flux.state

##########################
# Set up neural networks #
##########################

# Define Inputs #
#---------------#
const X_INPUTS = [:k, :z, :hli, :hle]
const LOC_INPUTS = [
    :A, :α, :δ,
    #:q, :ρ,
    #:Π, :H_bar, :elasticity,
    :A_bar, :α_bar, :δ_bar,
    :A_g, :α_g, :δ_g
]

const N_X_INPUTS = length(X_INPUTS)
const N_LOC_INPUTS = length(LOC_INPUTS)

get_n_V_inputs(n_shock_inputs) = N_X_INPUTS + N_LOC_INPUTS + n_shock_inputs
#get_n_price_inputs(n_shock_inputs) = n_shock_inputs

# Normalize Inputs and Outputs #
#------------------------------#
const X_MEAN = Float32[1492, 857, 4, 42]
const X_STD = Float32[2380.6582, 789.7718, 5.167285, 48.50662]
const LOC_INPUTS_MEAN = Float32[1, 3.6681085, 2.9920568, 1.0, 3.6681085, 2.9920568, -0.0138037, -0.011541073, 1.3024331]
const V_MEAN = 26f0
const V_STD = 4f0

const PRICE_MEAN = Float32[86; 36;;]
const PRICE_STD = Float32[78; 43;;]

"Keep this as it is: do not make it in-place."
normalize_x(x_ind, x) = @. (x - X_MEAN[x_ind]) / X_STD[x_ind]
normalize_loc_input(loc_input_ind, loc_input) = loc_input .- LOC_INPUTS_MEAN[loc_input_ind]

normalize_V!(V) = (V .-= V_MEAN; V ./= V_STD)
denormalize_V!(V) = (V .*= V_STD; V .+= V_MEAN)
normalize_price!(V) = (V .-= PRICE_MEAN; V ./= PRICE_STD)
denormalize_price!(V) = (V .*= PRICE_STD; V .+= PRICE_MEAN)

# Construct Neural Network Inputs #
#---------------------------------#
"""
Get a large array of inputs for the neural network. One for each state.
I feed in a normalized version of the actualy state values, but this is not strictly
necessary. I could have done a grid for each state variable. All we need is for the
neural net to tell the difference.
"""
function initialize_V_input!(V_input_arr)
    for (i, input) in enumerate([WEALTH_GRID, Z_GRID[1], H_LIVE_GRID, H_LET_GRID])
        V_input_arr[i,:,:,:,:,:] .= normalize_x(i, input)
    end
    return V_input_arr
end

as_input_matrix(arr) = reshape(arr, size(arr,1), prod(size(arr)[2:end]))
as_input_array(mat) = reshape(mat, size(mat,1), STATE_IDXs...)
as_price_input_array(mat) = reshape(mat, (size(mat,1),1,1,1,1,N_LOC))

get_V_input_arr_empty(n_V_inputs) = zeros(Float32, n_V_inputs, STATE_IDXs...)
get_V_input_arr(n_V_inputs) = initialize_V_input!(get_V_input_arr_empty(n_V_inputs))
get_V_input_mat(n_V_inputs) = as_input_matrix(get_V_input_arr(n_V_inputs))

# Apply Climate Path to Inputs #
#------------------------------#
function get_price_input(loc_grid::LocGrid, shock_hist::AbstractVector)
    price_input_mat = zeros(Float32, N_LOC_INPUTS+length(shock_hist), N_LOC)
    price_input_arr = as_price_input_array(price_input_mat)
    for (i, inputname) in enumerate(LOC_INPUTS)
        price_input_arr[i,:,:,:,:,:] .= normalize_loc_input(i, getproperty(loc_grid, inputname))
    end
    price_input_mat[N_LOC_INPUTS+1:end,:] .= shock_hist
    return price_input_mat
end

function add_loc_data_to_V_input!(V_input_mat::AbstractMatrix, loc_grid::LocGrid)
    V_input_arr = as_input_array(V_input_mat)
    for (i, inputname) in enumerate(LOC_INPUTS)
        V_input_arr[4+i,:,:,:,:,:] .= normalize_loc_input(i, getproperty(loc_grid, inputname))
    end
end

function add_shock_hist_to_V_input!(V_input_mat::AbstractMatrix, shock_hist::AbstractVector)
    # No need to normalize here because the two shock_hist inputs (SST and ε_m) are already
    # in the appropriate range.
    V_input_mat[N_X_INPUTS+N_LOC_INPUTS+1:end,:] .= shock_hist
    return V_input_mat
end

function apply_climate_to_V_input!(V_input_mat, loc_grid::LocGrid, shock_hist::AbstractVector)
    add_loc_data_to_V_input!(V_input_mat, loc_grid)
    add_shock_hist_to_V_input!(V_input_mat, shock_hist)
    return V_input_mat
end

function apply_climate_to_V_input_path!(V_input_path, loc_grid_path::LocGridPath, climpath::ClimatePath)
    shock_hist = vec(climpath.ε_m)
    for (deci, V_input) in enumerate(V_input_path)
        apply_climate_to_V_input!(V_input, LocGrid(loc_grid_path, deci), shock_hist[1:deci])
    end
    return V_input_path
end

function apply_climate_to_V_input_ss!(V_input_ss, loc_grid::LocGrid, climate::ClimateState)
    return apply_climate_to_V_input!(V_input_ss, loc_grid, [climate.ΔSST])
end

function apply_climate_to_V_inputs!(neural_nets, loc_grid_path::LocGridPath, climpath::ClimatePath)
    (;V_input_path, V_input_ss) = neural_nets
    n_dec = length(V_input_path)

    apply_climate_to_V_input_path!(V_input_path, loc_grid_path, climpath)
    apply_climate_to_V_input_ss!(V_input_ss, LocGrid(loc_grid_path, n_dec), climpath[end])
    return neural_nets
end

# Construct Output Arrays #
#-------------------------#
function make_nondecreasing!(v::AbstractVector)
    for i in 2:length(v)
        v[i] = max(v[i], v[i-1])
    end
    return v
end

function make_V_nondecreasing!(V::AbstractArray)
    for idx in CartesianIndices((N_Z, N_Hli, N_Hle, N_LOC))
        @views make_nondecreasing!(V[:,idx])
    end
    return V
end

get_normalized_V_mat(V) = normalize_V!(copy(reshape(V, 1, length(V))))
get_price_mat(loc_grid) = vcat(vec(loc_grid).q', vec(loc_grid.ρ)')
get_normalized_price_mat(loc_grid) = normalize_price!(get_price_mat(loc_grid))

function get_V_guess(V_nn, V_input)
    V_mat = V_nn(V_input)
    V = reshape(V_mat, STATE_IDXs)
    denormalize_V!(V)
    return make_V_nondecreasing!(V)
end
function get_price_guess(price_nn, price_input)
    price_output = price_nn(price_input)
    denormalize_price!(price_output)
    return max.(price_output, 1f0)
end

function guess_prices!(loc_grid::LocGrid, SST_hist::AbstractVector, price_nn_vec::AbstractVector)
    #price_input = get_price_input(loc_grid, SST_hist)
    price_guesses = get_price_guess.(price_nn_vec, Ref(SST_hist))
    vec(loc_grid.q) .= getindex.(price_guesses, 1)
    vec(loc_grid.ρ) .= getindex.(price_guesses, 2)
    return loc_grid
end

function guess_V_start(V_start_nn, loc_grid, shock_hist, V_input_mat)
    apply_climate_to_V_input!(V_input_mat, loc_grid, shock_hist)
    return get_V_guess(V_start_nn, V_input_mat)
end

# Define Neural Networks #
#------------------------#
get_V_start_nn(n_inputs) = Chain(
    Dense(n_inputs => 32, tanh),
    Dense(32 => 4, tanh),
    Flux.Bilinear(4 => 1),
)

get_price_nn(n_inputs) = Chain(
    Dense(n_inputs => 32, tanh),
    Dense(32 => 4, tanh),
    Dense(4 => 2, elu),
)

function get_V_start_nn_path(n_V_inputs_vec)
    V_nn_mat = [get_V_start_nn(n) for agei=1:N_AGE, n=n_V_inputs_vec]
    return reshape(V_nn_mat, 1,1,1,1,1,N_AGE,1,length(n_V_inputs_vec))
end

function get_price_nn_path(n_price_inputs_vec)
    p_nn_mat = [get_price_nn(n) for loci=1:N_LOC, n=n_price_inputs_vec]
    return reshape(p_nn_mat, 1,1,1,1,N_LOC,1,1,length(n_price_inputs_vec))
end

get_V_nn_period_vec(nn_path, deci) = nn_path[1,1,1,1,1,:,1,deci]
get_V_nn(nn_path, agei, deci) = get_V_nn_period_vec(nn_path, deci)[agei]
get_price_nn_period_vec(nn_path, deci) = nn_path[1,1,1,1,:,1,1,deci]
get_price_nn(nn_path, loci, deci) = get_price_nn_period_vec(nn_path, deci)[loci]

function get_neural_nets(n_dec)
    n_V_inputs_vec = get_n_V_inputs.(1:n_dec)

    # Transition Path
    V_start_nn_path = get_V_start_nn_path(n_V_inputs_vec)
    price_nn_path = get_price_nn_path(1:n_dec)
    V_input_path = get_V_input_mat.(n_V_inputs_vec)
    V_opt_path = Flux.setup.(Ref(Momentum(0.001)), V_start_nn_path)
    price_opt_path = Flux.setup.(Ref(Descent(1e-4)), price_nn_path)

    # Steady State
    V_start_ss_nn_vec = [get_V_start_nn(get_n_V_inputs(1)) for agei=1:N_AGE]
    price_ss_nn_vec = [get_price_nn(1) for loci=1:N_LOC]
    V_input_ss = get_V_input_mat(get_n_V_inputs(1))
    V_ss_opt = Flux.setup.(Ref(Momentum(0.001)), V_start_ss_nn_vec)
    price_ss_opt_vec = Flux.setup.(Ref(Descent(1e-4)), price_ss_nn_vec)
    
    return neural_nets = (;
        # Transition Path
        V_start_nn_path,
        price_nn_path,
        V_input_path,
        V_opt_path,
        price_opt_path,
        # Steady State
        V_start_ss_nn_vec,
        price_ss_nn_vec,
        V_input_ss,
        V_ss_opt,
        price_ss_opt_vec,
    )
end

function save_state(neural_nets)
    #V_state = Flux.state.(neural_nets.V_start_nn_path)
    #p_state = Flux.state.(neural_nets.price_nn_path)
    #V_ss_state = Flux.state.(neural_nets.V_start_ss_nn_vec)
    #p_ss_state = Flux.state.(neural_nets.price_ss_nn)
    V_state = neural_nets.V_start_nn_path
    p_state = neural_nets.price_nn_path
    V_ss_state = neural_nets.V_start_ss_nn_vec
    p_ss_state = neural_nets.price_ss_nn_vec

    nn_state = (;V_state, p_state, V_ss_state, p_ss_state)
    jldsave("nn_state.jld"; nn_state...)
end

#######################################
# Generate Guesses From Location Data #
#######################################

# Steady State #
#--------------#
function guess_and_apply_prices_ss!(period_data::PeriodData, shock_hist, price_ss_nn_vec)
    (;loc_grid) = period_data
    guess_prices!(loc_grid, shock_hist, price_ss_nn_vec)
    fill_in_q_last_steady_state!(period_data)
    precompute!(period_data, params)
    return period_data
end

function iterate_V_backwards_neural_net_ss!(age_data::AgeData, params::Params, V_start_nn_next, V_input_mat)
    (;agei, loc_grid) = age_data

    V_next = agei == N_AGE ? get_V_bequest!(age_data, loc_grid, params) : get_V_guess(V_start_nn_next, V_input_mat)
    solve_period!(age_data, V_next, params)
    return age_data
end

function iterate_V_backwards_neural_net_ss!(period_data::PeriodData, params::Params, V_start_ss_nn_vec, V_input_mat)
    for agei in reverse(1:N_AGE)
        V_start_nn_next = agei == N_AGE ? nothing : V_start_ss_nn_vec[agei+1]
        iterate_V_backwards_neural_net_ss!(AgeData(period_data, agei), params, V_start_nn_next, V_input_mat)
    end
    return period_data
end

# Transition Path #
#-----------------#
function guess_and_apply_prices!(path_data::PathData, climpath::ClimatePath, neural_nets, n_dec::Int)
    (;loc_grid_path) = path_data
    (;price_nn_path) = neural_nets
    shock_hist = climpath.ε_m

    for deci in 1:n_dec
        loc_grid = LocGrid(loc_grid_path, deci)
        guess_prices!(loc_grid, shock_hist[1:deci], get_price_nn_period_vec(price_nn_path, deci))
    end

    fill_in_q_last!(loc_grid_path)
    precompute!(path_data, params)
    return path_data
end

function guess_V_next!(V_start_nn_next, V_input_next, climate_params::ClimateParams)
    (;σ_m) = climate_params
    ε_m_save = copy(V_input_next[end,:]) # Can't be too careful

    V_next = zeros(Float32, STATE_IDXs)
    for ε_quantile in 0.1:0.1:0.9
        ε_m = quantile(Normal(0, σ_m), ε_quantile)
        V_input_next[end,:] .= ε_m
        V_next .+= get_V_guess(V_start_nn_next, V_input_next)
    end

    V_input_next[end,:] .= ε_m_save
    return V_next ./= 9
end

function iterate_V_backwards_neural_net!(
        age_data::AgeData, params::Params, V_start_nn_next, V_input_next, climate_params::ClimateParams
    )
    V_next = guess_V_next!(V_start_nn_next, V_input_next, climate_params)
    solve_period!(age_data, V_next, params)
    return age_data
end

function iterate_V_backwards_neural_net!(
        period_data::PeriodData, params::Params, V_start_nn_next_vec, V_input_next, climate_params::ClimateParams
    )
    for (agei, V_start_nn_next) in enumerate(V_start_nn_next_vec)
        age_data = AgeData(period_data, agei)
        iterate_V_backwards_neural_net!(age_data, params, V_start_nn_next, V_input_next, climate_params)
    end
    return period_data
end

function iterate_V_backwards_neural_net!(path_data::PathData, params::Params, climate_params::ClimateParams, neural_nets)
    (;n_dec) = path_data
    (;V_start_nn_path, V_input_path, V_start_ss_nn_vec, V_input_ss) = neural_nets

    # Iterate backwards for final period
    global deci_last = n_dec
    iterate_V_backwards_neural_net_ss!(PeriodData(path_data, n_dec), params, V_start_ss_nn_vec, V_input_ss)

    # Iterate backwards for mid-transition periods
    for deci in reverse(1:n_dec-1)
        global deci_last = deci
        period_data = PeriodData(path_data, deci)
        V_nn_next_vec = get_V_nn_period_vec(V_start_nn_path, deci+1)
        iterate_V_backwards_neural_net!(
            period_data, params, V_nn_next_vec, V_input_path[deci+1], climate_params
        )
    end
    return path_data
end


#########################
# Train Neural Networks #
#########################

"""
Train the neural net for V_start.
Note: V_input_mat must be updated with current location and climate data first.
"""
function train_V_start_nn!(V_start_nn, V_input_mat, opt, ad::AgeData, V_losses=Float32[])
    # This is the V_start implied by the guess of V_next from the neural net.
    V_output_target = get_normalized_V_mat(ad.V_price)
    #test_cols = rand(1:size(V_input_mat,2), 1000)
    #data = [(V_input_mat[:,test_cols], V_output_target[:,test_cols])]
    data = [(V_input_mat, V_output_target)]

    Flux.train!(V_start_nn, data, opt) do m, x, y
        error = Flux.mse(m(x), y)
        ChainRules.ignore_derivatives() do
            push!(V_losses, error)
        end
        return error
    end
    return V_start_nn
end

function train_V_start_nn!(V_start_nn_vec, V_input_mat, V_opt_vec, pd::PeriodData)
    V_losses = Float32[]
    for (agei, V_start_nn) in enumerate(V_start_nn_vec)
        ad = AgeData(pd, agei)
        train_V_start_nn!(V_start_nn, V_input_mat, V_opt_vec[agei], ad, V_losses)
    end
    return V_losses
end

function train_V_start_nn!(neural_nets, path_data::PathData, deci::Int)
    (;V_start_nn_path, V_input_path, V_opt_path) = neural_nets
    return train_V_start_nn!(
        get_V_nn_period_vec(V_start_nn_path, deci),
        V_input_path[deci],
        get_V_nn_period_vec(V_opt_path, deci),
        PeriodData(path_data, deci),
    )
end

function train_V_start_ss_nn!(neural_nets, path_data::PathData)
    (;n_dec) = path_data
    (;V_start_ss_nn_vec, V_input_ss, V_ss_opt) = neural_nets
    return train_V_start_nn!(V_start_ss_nn_vec, V_input_ss, V_ss_opt, PeriodData(path_data, n_dec))
end

function train_V_start_nn!(neural_nets, path_data::PathData)
    (;n_dec) = path_data
    V_losses = train_V_start_nn!.(Ref(neural_nets), Ref(path_data), 1:n_dec)
    return reduce(hcat, V_losses)
end

get_excess_demand_mat(pd, params) = vcat(transpose.(vec.(get_excess_demand(pd, params)))...)

function train_price_nn!(price_nn, shock_hist, excess_demand, opt)
    loss = Ref(0f0)

    data = [(shock_hist, excess_demand)]
    Flux.train!(price_nn, data, opt) do m, shock_hist, excess_demand
        nn_out = m(shock_hist)
        error = sum(-nn_out .* (excess_demand .* sign.(excess_demand)))
        ChainRules.ignore_derivatives() do
            loss[] = error
        end
        return error
    end
    return loss[]
end

function train_price_nn_period!(nn_vec, opt_vec, period_data::PeriodData, shock_hist, params::Params)
    excess_demand_mat = get_excess_demand_mat(period_data, params)

    return train_price_nn!.(nn_vec, Ref(shock_hist), eachcol(excess_demand_mat), opt_vec)
end

function train_price_nn_period!(neural_nets, path_data::PathData, climpath::ClimatePath, deci::Int, params::Params)
    (;price_nn_path, price_opt_path) = neural_nets

    return train_price_nn_period!(
        get_price_nn_period_vec(price_nn_path, deci),
        get_price_nn_period_vec(price_opt_path, deci),
        PeriodData(path_data, deci),
        climpath.ε_m[1:deci],
        params,
    )
end

function train_price_nn_path!(neural_nets, path_data::PathData, climpath::ClimatePath, params::Params)
    (;n_dec) = path_data
    price_losses = train_price_nn_period!.(Ref(neural_nets), Ref(path_data), Ref(climpath), 1:n_dec, params)
    return reshape(price_losses, 1, n_dec)
end


##############
# Outer Loop #
##############

function make_climate_draw!(path_data::PathData, neural_nets, climparams::ClimateParams, params::Params; ε_m=nothing)
    (;n_dec, loc_grid_path) = path_data
    # Get climate draw
    #climpath = get_climate_path(climparams, n_dec, zeros(Float32, n_dec))
    climpath = get_climate_path(climparams, n_dec, ε_m)
    # Apply climate to model
    apply_climate_to_loc_grid!(path_data, climpath, params, false)
    # Apply climate to input data
    apply_climate_to_V_inputs!(neural_nets, loc_grid_path, climpath)
    # Guess prices
    guess_and_apply_prices!(path_data, climpath, neural_nets, n_dec)
    # Guess values and simulate backward
    iterate_V_backwards_neural_net!(path_data, params, climparams, neural_nets)
    # Simulate forward
    solve_forward_transition!(path_data, params)
    return path_data, climpath
end

function guess_future_steady_state!(path_data::PathData, cs::ClimateState, neural_nets, params::Params)
    (;n_dec) = path_data
    (;V_start_ss_nn_vec, price_ss_nn_vec, V_input_ss) = neural_nets
    period_data = PeriodData(path_data, n_dec)
    guess_and_apply_prices_ss!(period_data, [cs.ΔSST[end]], price_ss_nn_vec)
    iterate_V_backwards_neural_net_ss!(period_data, params, V_start_ss_nn_vec, V_input_ss)
    simulate_forward_steady_state!(period_data, params)
    return period_data
end

function train_final_steady_state_nns!(neural_nets, pd::PeriodData, cs::ClimateState)
    (;V_start_ss_nn_vec, price_ss_nn_vec, V_input_ss, V_ss_opt, price_ss_opt_vec) = neural_nets
    V_losses = train_V_start_nn!(V_start_ss_nn_vec, V_input_ss, V_ss_opt, pd)
    price_loss = train_price_nn_period!(price_ss_nn_vec, price_ss_opt_vec, pd, [cs.ΔSST[end]], params)
    return V_losses, price_loss
end

"""
Global solution example loop:

params = Params()
climparams = ClimateParams(σ_m=1e-4)
#do_location_calibration(params)

n_dec = 3

# Preallocate path data
path_data = get_path_data_from_initial_steady_state(params, n_dec)

# Get neural nets
neural_nets = get_neural_nets(n_dec)
# Pre-train (to steady-state values)

# Train
i = 0
ε_m = zeros(Float32, 3)
while true
    i += 1
    path_data, climpath = make_climate_draw!(path_data, neural_nets, climparams, params; ε_m)
    V_losses = train_V_start_nn!(neural_nets, path_data)
    price_losses = train_price_nn_path!(neural_nets, path_data, climpath, params)
    println("V: ", mean(V_losses), " P: ", mean(price_losses))

    excess_demand_mat = get_excess_demand_mat(PeriodData(path_data, 1), params)
    @show excess_demand_mat
    @show vec(path_data.q)
    @show climpath.ε_m

    period_data = guess_future_steady_state!(path_data, climpath[end], neural_nets, params)
    V_losses, price_loss = train_final_steady_state_nns!(neural_nets, period_data, climpath[end])
    if i % 10 == 0
        save_state(neural_nets)
    end
end
"""