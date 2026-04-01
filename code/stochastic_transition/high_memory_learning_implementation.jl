
function get_∂err_∂θⱽ(vnet, gm, poparr, V_test)
    return ∂err_∂θⱽ = Flux.gradient(vnet) do vnet
        (;V_net_post, V_net_pre) = vnet
        V_pred = dropdims(V_net_post(V_net_pre(poparr) .+ gm).*20; dims=1)
        return Flux.mse(V_pred, V_test) / 29.6f0
    end |> only
end

function get_∂err_∂gm(vnet, gm, poparr, V_test)
    (;V_net_post, V_net_pre) = vnet

    err, ∂err_∂gm = Flux.withgradient(gm) do gm
        V_pred = dropdims(V_net_post(V_net_pre(poparr) .+ gm).*20; dims=1)
        return Flux.mse(V_pred, V_test) / 29.6f0
    end
    return err, only(∂err_∂gm)
end

function get_∂err_∂θᵛ_and_∂gm(vnet, gm, poparr, V_test)
    err, grads = Flux.withgradient(vnet, gm) do vnet, gm
        
        (;V_net_post, V_net_pre) = vnet
        V_pred = dropdims(V_net_post(V_net_pre(poparr) .+ gm).*20; dims=1)
        return Flux.mse(V_pred, V_test) / 29.6f0
    end
    ∂err_∂θⱽ = grads[1]
    ∂err_∂gm = grads[2]
    return (;err, ∂err_∂θⱽ, ∂err_∂gm)
end

