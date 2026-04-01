
##########################
# Generate Training Data #
##########################

# From Deterministic Simulation #
#-------------------------------#
"Prepare training data from one period of perfect-foresight transition path data."
function get_training_data_deterministic(pd_last::PeriodData, params_last::Params, pd::PeriodData, params::Params)
    Λ_start = get_Λ_start_next(pd_last, params_last)
    input = get_vnet_input(Λ_start, params)
    label = stack(pd.V_end[1:end-1])
    return input, label
end

"Prepare training data from perfect-foresight transition path data."
function get_training_data_path_deterministic(td::Vector{<:PeriodData}, pp::Vector{<:Params})
    T = length(td)
    return [get_training_data_deterministic(td[t-1], pp[t-1], td[t], pp[t]) for t=2:T-2]
    # Using T-2 instead of T-1 as the last period only because that's what I do for the stochastic case.
end

"""
Compute predicted expected value function at start of next period, by simulating forward one period using vnet,
then iterating backward one period, then taking expectations over each possible SST state next period.
"""
function get_EV_start_next_by_forward_simulation_lookahead(pd_next, pd, params, vnet, params_next; V_end_next=nothing, kwargs...)

    # Save original scenario_ind_next
    scenario_ind_next_save = params_next.scenario_ind

    # For each possible scenario_ind_next, simulate forward and iterate backward to get V_start_next,
    # then take expectations over each scenario_ind_next.
    get_V_start_next(scenario_ind) = simulate_forward_given_vnet!(pd_next, pd, params, vnet, params_next; scenario_ind, V_end=V_end_next, kwargs...).V_start
    weights = SST_TRANSMAT[params.scenario_ind, :]
    EV_start_next_pred = sum(get_V_start_next(scenario_ind)[2:end] .* weights[scenario_ind] for scenario_ind=1:N_SCENARIO)

    # Restore original scenario_ind_next
    params_next.scenario_ind = scenario_ind_next_save

    return EV_start_next_pred
end

# From Stochastic Simulation Via Lookahead #
#------------------------------------------#
function get_training_data_stochastic(pd_last::PeriodData, pd::PeriodData, pd_next::PeriodData, params_last::Params, params::Params, params_next::Params, vnet::VNet; predict_V_end_next=true, verbosity=0, t=nothing, scenario_ind, kwargs...)

    # Set scenario_ind for this period, either by drawing or using the provided value.
    params.scenario_ind = scenario_ind==:draw ? draw_scenario_ind(params_last.scenario_ind) : scenario_ind

    # Prepare the neural network input: Household state distribution at the end of the previous period,
    # and parameters in the current period.
    Λ_start = get_Λ_start_next(pd_last, params_last)
    input = get_vnet_input(Λ_start, params)

    # Simulate to the end of the current period
    simulate_forward_given_vnet!(pd, pd_last, params_last, vnet, params; scenario_ind=params.scenario_ind, verbosity=verbosity-1, kwargs...)

    # Prepare the neural network label: Predicted expected value function at start of next period,
    # computed by simulating forward another period, then iterating backward one period, then taking
    # expectations over each possible SST state next period.
    V_end_next = predict_V_end_next ? nothing : pd_next.V_end[1:end-1]
    V_end_pred = get_EV_start_next_by_forward_simulation_lookahead(pd_next, pd, params, vnet, params_next; V_end_next, verbosity=verbosity-1, kwargs...)
    label = stack(V_end_pred)

    verbosity >= 1 && println("Finished simulating forward t=$t, scenario_ind=$(params.scenario_ind)")

    return input, label
end

function get_training_data_path_stochastic(td::Vector{<:PeriodData}, pp::Vector{<:Params}, vnet::VNet; SST_path=nothing, kwargs...)
    T = length(td)

    SST_path = something(SST_path, draw_scenario_path(pp[1].scenario_ind, T))
    
    # For the last period used (t=T-2), don't predict V_end_next (t=T-1).
    # Instead, use the value implied by the steady state solution (t=T).
    # I don't generate training data for t=T-1 as well, even though I technically could,
    # because the V_end[T-1] is not a very good representation of the predicted value function.
    # Ideally, we should recompute the new steady state every time, or at least leave a buffer of
    # several periods between the end of the training data transition and the new steady state.
    training_data = [get_training_data_stochastic(td[t-1:t+1]..., pp[t-1:t+1]..., vnet; predict_V_end_next=(t!=T-2), t, scenario_ind=SST_path[t], kwargs...) for t=2:T-2]

    @assert all(pp[t].scenario_ind == SST_path[t] for t=1:T-2)
    return training_data
end

function get_training_data(td::Vector{<:PeriodData}, pp::Vector{<:Params}; vnet=nothing, stochastic::Bool, kwargs...)
    if stochastic
        return get_training_data_path_stochastic(td, pp, vnet; kwargs...)
    else
        return get_training_data_path_deterministic(td, pp)
    end
end


##############################
# Low-Memory Training Tricks #
##############################

"""
This awkward stack of functions exists purely to add one set of gradients to another, when
each is represented in the nested structure used by Flux for gradients.
"""
sum_recursive!(layer1::Nothing, layer2) = layer2
sum_recursive!(layer1::AbstractArray, layer2::AbstractArray) = layer1 .+= layer2
sum_recursive!(layer1, layer2)  = layer1 + layer2
sum_recursive!(layer1::NamedTuple{K}, layer2::NamedTuple{K}) where K = NamedTuple{K}(sum_recursive!.(values(layer1), values(layer2)))
sum_recursive!(layer1::Tuple{K}, layer2::Tuple{K}) where K = Tuple{K}(sum_recursive!.(layer1, layer2))
sum_recursive!(layer1::Tuple, layer2::Tuple) = Tuple(sum_recursive!.(layer1, layer2))

"""
For a given set of generalized moments gm and population moments popmom, compute the predicted
value of V_end, and compare with the given V_test. Automatically compute the derivative of the
error with respect to (1) the parameters of V_net_post, (2) gm, and (3) popmom.
"""
function get_∂err_∂θ_V_post_gm_and_popmom(vnet, gm, popmom, V_test; sample_hh_states=true, sample_size=100)

    n_hh_states = prod(size(popmom)[2:end])
    sample_inds = sample_hh_states ? sample(1:n_hh_states, sample_size; replace=false) : 1:n_hh_states
    
    err, grads = Flux.withgradient(vnet, gm, popmom) do vnet, gm, popmom
        (;V_net_post) = vnet

        # Sample household states
        popmom_mat = array_to_matrix(popmom)[:, sample_inds]
        V_test_vec = reshape(V_test, (n_hh_states, size(V_test,4)))[sample_inds, :] |> vec

        # Compute V_end prediction and error
        V_pred = vec(V_net_post(popmom_mat .+ gm).*20)
        return Flux.mse(V_pred, V_test_vec) / mean(V_test_vec)^2
    end
    return (;err, ∂err_∂θ_V_post=grads[1], ∂err_∂gm=grads[2], ∂err_∂popmom=grads[3])
end

function get_∂err_∂θᵍ(vnet, train_in, ∂err_∂gm)
    return ∂err_∂θᵍ = Flux.gradient(vnet) do vnet
        gms = get_generalized_moments(train_in, vnet)
        return gms⋅∂err_∂gm
    end |> only
end

function get_∂err_∂θ_V_pre(vnet, train_in, ∂err_∂popmoms)
    return ∂err_∂θ_V_pre = Flux.gradient(vnet) do vnet
        (;V_net_pre) = vnet
        pop_arrays = train_in.pop_arrays
        popmoms = stack([V_net_pre(pop_arr) for pop_arr in pop_arrays])
        return popmoms⋅∂err_∂popmoms
    end |> only
end

"""
Train model one location at a time to reduce VRAM usage.

Let θᵍ be the parameters of the network which computes the aggregate and
location-level generalized moments gms as a function of aggregate and
location-level states, including distributions of household states.

Let θⱽ be the parameters of the network which computes the value function V
as a function of idiosyncratic states and the generalized moments gms.

The trick is to compute the generalized moments gms only once, and take them
as given for the purposes of computing ∂err/∂θⱽ by autodiff. This can be computed
for each location separately, then summed together as you go using the rather awkward
function sum_recursive!. This avoids backwards iterating through the gm network
for each location.

To compute ∂err/∂θᵍ, we first compute ∂err/∂gms for each location separately.
Then, backpropagate through the gm network with the objective of gms⋅∂err/∂gms. The resulting
derivative will be equal to ∂gms/∂θᵍ⋅∂err/∂gms = ∂err/∂θᵍ.
"""
function train_model_low_memory!(vnet, train_in_cpu, train_out_cpu, opt_state; verbosity=0, t=nothing, locations=1:N_LOC, loc_sample_size=min(50, N_LOC), kwargs...)

    # Move training data to GPU
    train_in = gpu(train_in_cpu)
    train_out = gpu(train_out_cpu)
    (;pop_arrays) = train_in
    V_test_ageslices = eachslice(train_out; dims=5)

    # Compute generalized moments once
    gms = get_generalized_moments(train_in, vnet)
    gms_locslices = eachslice(gms; dims=5)

    # Initialize gradients and total loss
    ∂err_∂gms = zero(gms)
    ∂err_∂popmoms = gpu(zeros(Float32, (vnet.width, N_K, N_Z, N_H, N_AGE-1)))
    ∂err_∂θ_V_post = nothing
    tot_loss = 0f0

    # Set of locations to train on
    if locations == :sample
        locations = sample(1:N_LOC, loc_sample_size; replace=false)
    end

    # Compute ∂err/∂θⱽ and ∂err/∂gms one (location, age) at a time
    for agei in 1:N_AGE-1
        # Compute population moments once
        poparr = pop_arrays[agei]
        popmom = vnet.V_net_pre(poparr)

        # Get testing data slice for this age group
        V_test_age = V_test_ageslices[agei]
        V_test_locslices = eachslice(V_test_age; dims=4)

        if length(locations) > 50
            for loci in locations
                # Differentiate error 
                grads = get_∂err_∂θ_V_post_gm_and_popmom(vnet, gms_locslices[loci], popmom, V_test_locslices[loci]; kwargs...)
                
                tot_loss += grads.err/length(locations)
                ∂err_∂θ_V_post = sum_recursive!(∂err_∂θ_V_post, grads.∂err_∂θ_V_post)
                ∂err_∂gms[:,:,:,:,loci] .+= grads.∂err_∂gm
                ∂err_∂popmoms[:,:,:,:,agei] .+= grads.∂err_∂popmom

                verbosity >= 3 && @show loci
            end
        else
            grads = get_∂err_∂θ_V_post_gm_and_popmom(vnet, gms[:,:,:,:,locations], popmom, V_test_age[:,:,:,locations])
            tot_loss += grads.err
            ∂err_∂θ_V_post = sum_recursive!(∂err_∂θ_V_post, grads.∂err_∂θ_V_post)
            ∂err_∂gms[:,:,:,:,locations] .+= grads.∂err_∂gm
            ∂err_∂popmoms[:,:,:,:,agei] .+= grads.∂err_∂popmom
        end

        verbosity >= 2 && @show agei
    end

    # Convert to average relative loss
    tot_loss /= N_AGE - 1
    tot_loss = sqrt(tot_loss)

    # Compute ∂err/∂θᵍ
    ∂err_∂θᵍ = get_∂err_∂θᵍ(vnet, train_in, ∂err_∂gms)
    ∂err_∂θ_V_pre = get_∂err_∂θ_V_pre(vnet, train_in, ∂err_∂popmoms)
    @assert isnothing(∂err_∂θ_V_post.V_net_pre)
    
    # Report progress
    verbosity >= 1 && @show (tot_loss, t)

    # Update model parameters using accumulated gradients
    if !isnothing(opt_state)
        Flux.update!(opt_state, vnet, ∂err_∂θ_V_post)
        Flux.update!(opt_state, vnet, ∂err_∂θ_V_pre)
        Flux.update!(opt_state, vnet, ∂err_∂θᵍ)
    end

    return (;vnet, tot_loss)
end


########################
# Train Neural Network #
########################
"Train vnet on training data provided by get_train_set, a function which returns an iterable of training data."
function train_vnet!(vnet, get_train_set; lr=1e-4, epochs=100, batch_size=100, verbosity=0, locations=:sample, loc_sample_size=min(50, N_LOC), save_intermediate=true, save_suffix="_tmp", kwargs...)

    #schedule = Stateful(CosAnneal(λ0 = 1e-2, λ1 = 1, period = 10))
    # Keep track of the total loss
    @assert locations in [:sample, 1:N_LOC]
    lr /= locations == :sample ? loc_sample_size : N_LOC
    global loss_hist = Float32[]
    local tot_loss

    # Set up the optimizer
    opt = Flux.setup(AdamW(lr), vnet)
    best_loss = Inf32
    t_since_best_loss = 0

    
    # Training loop
    for epoch=1:epochs
        tot_loss = 0f0
        #Flux.adjust!(opt, lr*next!(schedule))
        train_set = get_train_set()

        # Train batch_size times over the full training set
        for i=1:batch_size
            for (t,data) in enumerate(train_set)
                train_in_cpu, train_out_cpu = data
                tot_loss += train_model_low_memory!(vnet, train_in_cpu, train_out_cpu, opt; verbosity=verbosity-1, t, locations, loc_sample_size, kwargs...).tot_loss
            end

            verbosity >= 1 && println("Batch $i/$batch_size completed.")
        end

        # Compute total loss and keep track of best loss
        tot_loss /= batch_size * length(train_set)
        push!(loss_hist, tot_loss)        
        t_since_best_loss = tot_loss < best_loss ? 0 : t_since_best_loss + 1
        best_loss = min(best_loss, tot_loss)
        
        # Reduce learning rate on plateau
        if t_since_best_loss > 5
            #lr = max(lr*0.75f0, 1f-7)
            #Flux.adjust!(opt, lr)
            #println("Reducing learning rate to $lr")
            #open(io->println(io, now(), " Reducing learning rate to $lr"), "training.log", "a")
            #t_since_best_loss = 0
        end

        # Verbose output
        out_string = "Epoch $epoch completed. Total loss: $tot_loss. Epochs since best loss: $t_since_best_loss."
        verbosity >= 1 && println(out_string)
        open(io->println(io, now(), ' ', out_string), "training.log", "a")

        # Save intermediate model
        save_intermediate && save_vnet(vnet; suffix=save_suffix)
    end

    return (;tot_loss, vnet)
end


#######################################
# Load and Save Neural Net Parameters #
#######################################

function load_vnet!(vnet; suffix="")
    Flux.loadmodel!(vnet, load(NEURAL_NET_PARAMS_DIR*"vnet_state_$(vnet.width)_$(N_LOC)loc$(suffix).jld2", "model_state"))
    return vnet
end

function save_vnet(vnet; suffix="")
    save(NEURAL_NET_PARAMS_DIR*"vnet_state_$(vnet.width)_$(N_LOC)loc$(suffix).jld2", "model_state", Flux.state(vnet))
end
