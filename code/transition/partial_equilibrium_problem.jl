

############
# Backward #
############

function iterate_V_backward_transition!(td::Vector{<:PeriodData}, pp::Vector{<:Params}; verbosity=0, pd_gpu=nothing)

    for t=reverse(2:length(td)-1)
        td[t].V_end[end] .= get_V_bequest(td[t].prices, pp[t])
        @assert td[t].q_last == td[t-1].q

        for agei=1:N_AGE-1
            td[t].V_end[agei] .= td[t+1].V_start[agei+1]
        end

        if ACTIVELY_DISPATCH_TO_GPU
            # Move data to GPU
            pd_g = isnothing(pd_gpu) ? to_gpu(td[t]) : to_gpu!(pd_gpu, td[t])
            params_g = to_gpu(pp[t])

            # Solve on GPU
            iterate_V_backward!(pd_g, gpu.(td[t].V_end), params_g)

            # Move output back to CPU
            to_cpu!(td[t], pd_g)
            to_cpu!(pp[t], params_g)
        else
            iterate_V_backward!(td[t], td[t].V_end, pp[t])
        end

        verbosity > 0 && println("V backward t=$t")
    end
    return td
end


###########
# Forward #
###########

function iterate_λ_forward_transition!(td::Vector{<:PeriodData}, pp::Vector{<:Params}; verbosity=0, pd_gpu=nothing)

    # Compute moments at endpoints
    H_D_1 = get_H_D(td[1])
    H_D_end = get_H_D(td[end])
    H_D = [fill(H_D_1, length(td)-1)..., H_D_end]
    H_rent_1 = get_H_rent(td[1], pp[1])
    H_rent_end = get_H_rent(td[end], pp[end])
    H_rent = [fill(H_rent_1, length(td)-1)..., H_rent_end]

    for t=2:length(td)-1
        td[t].λ_start[1] .= get_λ_born_based_on_V(td[t].V_start[1], pp[t])
        for agei=2:N_AGE
            td[t].λ_start[agei] .= td[t-1].λ_end[agei-1]
        end

        if ACTIVELY_DISPATCH_TO_GPU
            # Move data to GPU
            pd_g = isnothing(pd_gpu) ? to_gpu(td[t]) : to_gpu!(pd_gpu, td[t])
            params_g = to_gpu(pp[t])

            # Solve on GPU
            iterate_λ_forward!(pd_g, gpu.(td[t].λ_start), params_g)
            H_D_t = get_H_D(pd_g)
            H_rent_t = get_H_rent(pd_g, params_g)

            # Move output back to CPU
            to_cpu!(td[t], pd_g)
            to_cpu!(pp[t], params_g)
            H_D[t] = cpu(H_D_t)
            H_rent[t] = cpu(H_rent_t)
        else
            iterate_λ_forward!(td[t], td[t].λ_start, pp[t])

            # Compute moments
            H_D[t] = get_H_D(td[t])
            H_rent[t] = get_H_rent(td[t], pp[t])
        end
        
        verbosity > 0 && println("λ forward t=$t")
    end

    assert_approx(pp[1].H_S, get_H_S(td[1].q, pp[1]))
    assert_approx(pp[end].H_S, get_H_S(td[end].q, pp[end]))
    for t=2:N_AGE-1
        assert_approx(pp[t].H_S, get_H_S(td[t].q, pp[t], pp[t-1].H_S))
    end
    return (;td, H_rent, H_D)
end


############
# As Stage #
############

function solve_given_prices!(td::Vector{<:PeriodData}, pp::Vector{<:Params}; ρ_path=getproperty.(td, :ρ), q_path=getproperty.(td, :q), verbosity=0, pd_gpu=nothing)
    
    T = length(td)
    @assert ρ_path[1] == td[1].ρ
    @assert q_path[1] == td[1].q
    @assert ρ_path[end] == td[end].ρ
    @assert q_path[end] == td[end].q

    for t=2:T-1
        set_prices_and_params!(td[t], pp[t]; ρ=ρ_path[t], q=q_path[t], q_last=q_path[t-1], H_S_prior=pp[t-1].H_S)
    end

    iterate_V_backward_transition!(td, pp; verbosity=verbosity, pd_gpu)
    (;td, H_rent, H_D) = iterate_λ_forward_transition!(td, pp; verbosity=verbosity, pd_gpu)

    assert_approx(H_rent[1], get_H_rent(td[1], pp[1]))
    assert_approx(H_rent[2], get_H_rent(td[2], pp[2]))
    assert_approx(H_D[1], get_H_D(td[1]))
    assert_approx(H_D[2], get_H_D(td[2]))

    return (;td, H_rent, H_D)
end
