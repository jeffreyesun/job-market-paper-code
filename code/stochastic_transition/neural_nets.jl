
"""
Define neural networks to approximate V_start and prices.
First, compute ``generalized moments'' for each location, summarizing the environment and population distribution in that location and across space.
Next, use these generalized moments, together with idiosyncratic state, to predict V_end.
"""

const POP_ARRAYS = [stack([FLOAT[idx[1]/N_K, idx[2]/N_Z, idx[3]/N_H, agei/N_AGE] for idx=CartesianIndices(STATE_IDXs[1:end-1])]) for agei=1:N_AGE-1]
const ID_NLOC = collect(I(N_LOC))

##########################
# Neural Net Definitions #
##########################


get_ff_net(ins, outs; width=512) = Chain(
    Dense(ins => width, elu),
    LayerNorm(width),
    Dense(width => width, elu),
    LayerNorm(width),
    Dense(width => width, elu),
    LayerNorm(width),
    Dense(width => width, elu),
    LayerNorm(width),
    Dense(width => width, elu),
    LayerNorm(width),
    Dense(width => outs, elu),
)
# ins: A, α, δ, H_S_prior, q_last, elasticity, Π, one-hot dummy

@kwdef struct VNet{E, P, L, G, V1, V2}
    width::Int = 512
    env_net::E = get_ff_net(9 + N_LOC, width; width)
    pop_agg_net::P = get_ff_net(4, width; width)
    loc_agg_net::L = get_ff_net(width + N_LOC, width; width) #4 + N_LOC, 64)
    gm_net_post::G = get_ff_net(width, width; width)

    V_net_pre::V1 = get_ff_net(4, width; width)
    V_net_post::V2 = get_ff_net(width, 1; width)
end


#######################
# Generalized Moments #
#######################

function get_pop_gm(pop_arr, λ_start, pop_agg_net)
    pop_mat = reshape(pop_arr, (4, N_K*N_Z*N_H))
    λ_start_mat = reshape(λ_start, (N_K*N_Z*N_H, N_LOC))
    return pop_agg_net(pop_mat) * λ_start_mat ./ sum(λ_start_mat) # This is a matmul
end

"""
Get a generalized moment for each location which summarizes information about the local and aggregate state.
`env_gms` contains local information about the non-population state (environment, housing stock, previous prices)
`pop_gms` contains information about the population distribution in each location.
`loc_gms` sums the above two types of information about the local state.
`agg_gm` contains information about the distribution of local states across locations.
Each column of `gms` contains information about that location *and* the aggregate state.
"""
function get_generalized_moments(loc_mat, λ_start_surviving, vnet::VNet, pop_arrays)
    (;env_net, pop_agg_net, loc_agg_net, gm_net_post) = vnet

    # Local non-population information
    env_gms = env_net(loc_mat)

    # Local population information
    pop_gms = sum([get_pop_gm(pop_arr, λ_start, pop_agg_net) for (pop_arr, λ_start)=zip(pop_arrays, λ_start_surviving)])

    # Aggregate information
    loc_gms = env_gms + pop_gms
    agg_gm = mean(loc_agg_net(vcat(loc_gms, ID_NLOC)); dims=2)

    # Information combining local and aggregate state
    gms_pre = reshape(loc_gms .+ agg_gm, (size(loc_gms,1), 1,1,1,N_LOC))
    return gm_net_post(gms_pre)
end
function get_generalized_moments(train_in, vnet::VNet)
    (;loc_mat, λ_start_surviving, pop_arrays) = train_in
    return get_generalized_moments(loc_mat, λ_start_surviving, vnet, pop_arrays)
end

"Get the predicted V_end, given beginning-of-period data."
function (vnet::VNet)(loc_mat, λ_start_surviving, pop_arrays, agei)
    (;V_net_pre, V_net_post) = vnet

    if isa(agei, Int)
        agei = [agei]
    end

    # Compute generalized moments summarizing the environment and population distribution
    # as it affects V_end in each location
    gms = get_generalized_moments(loc_mat, λ_start_surviving, vnet, pop_arrays)

    # Convert generalized moments to V_end predictions nonlinearly, excluding the last age group,
    # who will receive the bequest motive
    V_pred = stack(stack(V_net_post(V_net_pre(pop_arr) .+ gms[:,1,1,1,loci]).*20 for loci=1:N_LOC) for pop_arr=pop_arrays[agei])
    
    return dropdims(V_pred; dims=1)
end
(vnet::VNet)(inputs, agei=1:N_AGE-1) = vnet(inputs.loc_mat, inputs.λ_start_surviving, inputs.pop_arrays, agei)

Flux.@layer VNet
