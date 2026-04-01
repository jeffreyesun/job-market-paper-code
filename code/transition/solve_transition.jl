
# Create Transition Data Container #
#----------------------------------#
function get_transition_data(pp::Vector{<:Params}; q=nothing, ρ=nothing, q_last=nothing)
    T = length(pp)
    q = something(q, fill(nothing, T))
    ρ = something(ρ, fill(nothing, T))
    q_last = something(q_last, fill(nothing, T))

    td = [PeriodData(get_test_prices(), pp[t]; q=q[t], ρ=ρ[t], q_last=q_last[t]) for t=1:T]
end

# Solve Initial and Final Steady States #
#---------------------------------------#
function solve_steady_state_endpoints!(td, pp; kwargs...)
    solve_steady_state!(td[1], pp[1]; kwargs...)
    solve_steady_state!(td[end], pp[end]; kwargs...)
    return td
end

function get_transition_data_with_steady_state_endpoints(pp::Vector{<:Params}; q=nothing, ρ=nothing, q_last=nothing, kwargs...)
    td = get_transition_data(pp; q=q, ρ=ρ, q_last=q_last)
    
    return td = solve_steady_state_endpoints!(td, pp; kwargs...)
end

# Solve Full Transition Given Solved Initial and Final Steady States #
#--------------------------------------------------------------------#
function solve_transition_given_solved_endpoints!(td::Vector{<:PeriodData}, pp::Vector{<:Params}; update_speed=10, rtol=1e-4, verbosity=0)

    T = length(td)
    transition_price_err = Inf
    iterations = 0
    
    pd_gpu = ACTIVELY_DISPATCH_TO_GPU ? to_gpu(td[1]) : nothing

    while transition_price_err > rtol
        (;td, H_rent, H_D) = solve_given_prices!(td, pp; verbosity=verbosity-1, pd_gpu)

        H_S = getproperty.(pp, :H_S)

        H_rent_err = [(H_rent[t] - H_D[t])./H_D[t] for t=1:T]
        H_D_err = [(H_D[t] - H_S[t])./H_S[t] for t=1:T]

        update_speed_cos = update_speed * cos(iterations/100)^2
        
        transition_price_err = rms(stack(H_rent_err)) + rms(stack(H_D_err))
        
        verbosity > 0 && @printf "Transition H_rent_err %e, H_D_err %e, transition_price_err %e\n" rms(stack(H_rent_err)) rms(stack(H_D_err)) transition_price_err
        
        iterations += 1
        if iterations == 10000
            @warn "Solver stuck at tolerance $rtol after $iterations iterations. Returning current prices."
            break
        end

        transition_price_err <= rtol && break

        for t=2:T-1
            qt = td[t].q
            ρt = td[t].ρ
            @. qt += update_speed_cos * H_rent_err[t] * qt/ρt*0.5
            @. qt += update_speed_cos * H_D_err[t] * 1
            @. ρt += update_speed_cos * H_rent_err[t] * 1
        end
    end

    return td
end

# Solve Full Transition From Scratch #
#------------------------------------#
function solve_transition(pp; verbosity=2, q=nothing, q_last=nothing, ρ=nothing, update_speed=10)

    verbosity >= 1 && println("Generating TransitionData container...")
    td = get_transition_data(pp; q, q_last, ρ)

    verbosity >= 1 && println("Solving transition path at endpoints...")
    td = solve_steady_state_endpoints!(td, pp; verbosity=verbosity-1, update_speed=1, rtol=5e-5)

    verbosity >= 1 && println("Solving full transition path...")
    td = solve_transition_given_solved_endpoints!(td, pp; verbosity=verbosity-1, update_speed)

    return (;td, pp)
end
