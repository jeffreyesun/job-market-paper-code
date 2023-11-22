
"""
A state machine-based parameter scheduler, JeffreysReallyBadScheduler.
It's really bad.
This is a tool for tuning the stepsize of the optimizer to speed up the
convergence of the solution algorithm, without overshooting.
Easier said than done.

I organize this file in reverse order, so that functions call functions below them.
"""

# Infer Convergence State and Dispatch #

"""
Update the stepsize used by the optimizer and decide whether to exit.

I attempt to divide the (optimization surface)x(stepsize) space into
useful regions by observables, and then dispatch on that inferred region.

The basic idea is: if convergence is slow but stable, we increase the stepsize.
If convergence is oscillating, decrease the stepsize.
"""
function next_stepsize!(s::JeffreysReallyBadScheduler, error, verbose=false)
    verbose && @show error
    error_min_last = s.error_min

    update_progress_counters!(s, error)
    isnan(s.error_last) && return s.stepsize

    change_over_best = (s.error - error_min_last) / s.error_min
    # Error derivative is unreliable. It can cause premature exit if
    # we overshoot by exactly enough that the error is unaffected.
    #error_derivative = (s.error - s.error_last) / s.stepsize

    if s.error <= s.ε
        return stop!(s, verbose, "Stopping due to convergence")
    end

    return s.stepsize #TODO Take out

    distance_from_best = change_over_best <= 0 ? :at_global_best : :away_from_global_best
    change_over_best == 0 && (distance_from_best = :stuck_at_global_best)

    #print(distance_from_best, "\t")
    return next_stepsize!(s, verbose, Val(distance_from_best))
end

function next_stepsize!(s::JeffreysReallyBadScheduler, verbose, ::Val{:at_global_best})
    # Reset progress counters
    s.error_min = s.error

    # If you're making progress slowly and reliably, increase stepsize
    if s.iterations_with_slow_progress >= s.max_iterations_with_slow_progress
        return increase_stepsize!(s, verbose, "Increased stepsize due to slow progress")
    else
        return s.stepsize
    end
end

function next_stepsize!(s::JeffreysReallyBadScheduler, verbose, ::Val{:stuck_at_global_best})
    # Current error is exactly equal to the best previous error.
    # If you're locally stuck too, stop
    if s.error == s.error_last
        return stop!(s, verbose, "Stopping due to local and global lack of progress")
    end

    # If you're oscillating, decrease stepsize. Otherwise, keep going
    if s.iterations_oscillating >= 3
        return decrease_stepsize!(s, verbose)
    else
        return next_stepsize!(s, verbose, Val(:at_global_best))
    end
end

function next_stepsize!(s::JeffreysReallyBadScheduler, verbose, ::Val{:away_from_global_best})
    # So you're away from the best error somehow. Local progress could be:
    # 1. positive: you're making your way back
    # 2. zero: you're stuck
    # 3. negative: you're moving away from the best error
    @assert s.error > s.error_min

    # 1. If you're making your way back, keep going
    if s.iterations_with_local_progress >= 1
        # Increase stepsize if progress is slow and steady
        if s.iterations_with_slow_progress >= s.max_iterations_with_slow_progress
            return increase_stepsize!(s, verbose, "Increased stepsize due to slow progress away from global best")
        else
            return s.stepsize
        end
    end

    # 2. If you're locally stuck, decrease stepsize or stop
    if s.error == s.error_last && s.iterations_without_local_progress > 3
        return get_locally_unstuck(s, verbose)
    end

    # 3. If you're neither making progress nor stuck, we have to try to diagnose
    return next_stepsize!(s, verbose, Val(:moving_away_from_global_best))
end

function next_stepsize!(s::JeffreysReallyBadScheduler, verbose, ::Val{:moving_away_from_global_best})
    # So you are away from the global best and moving away from it. A number of things could be wrong.
    # We need to detect when to decrease stepsize.
    # I propose we detect how long we have been oscillating or getting worse
    # and decrease stepsize if it's been too long.

    # If local error is exactly what it just was, stop
    if s.error == s.error_last
        if s.iterations_without_local_progress > 3
            return stop!(s, verbose, "Stopping due to local lack of progress")
        end
    end

    # If you've been away from the global best for too long, stop
    if s.iterations_without_global_progress > 30
        return stop!(s, verbose, "Stopping due to global lack of progress")
    end
 
    # If you're just moving away from the best error, eventually decrease stepsize
    if s.iterations_without_local_progress > s.max_iterations_without_local_progress
        return decrease_stepsize!(s, verbose, "Decreasing stepsize due to lack of local progress")
    end

    #TODO Take out
    return s.stepsize

    # If you've been oscillating, decrease stepsize
    if s.iterations_oscillating >= 5
        return decrease_stepsize!(s, verbose, "Decreasing stepsize due to oscillation")
    end

    if s.iterations_without_consistent_progress > 20
        return decrease_stepsize!(s, verbose, "Decreasing stepsize due to lack of consistent progress")
    end

    # If you're moving away from the best error very quickly, decrease stepsize
    if s.error > s.error_last*1.5
        return decrease_stepsize!(s, verbose, "Decreasing stepsize due to rapid increase in error")
    end

    # If you've been away from the global best for too long, stop
    if s.iterations_without_global_progress > s.max_iterations_without_global_progress
        return get_locally_unstuck(s, verbose)
    end

    return s.stepsize
end

function get_locally_unstuck(s, verbose)
    if s.n_stepsize_decreased - s.n_stepsize_increased >= 6
        return stop!(s, verbose, "Stopping AWAY from global optimum due to local lack of progress")
    else
        return decrease_stepsize!(s, verbose, "Decreasing stepsize due to getting locally stuck")
    end
end

get_change_over_last(s::JeffreysReallyBadScheduler) = (s.error - s.error_last) / s.error_last

###################
# Update Stepsize #
###################

function decrease_stepsize!(s::JeffreysReallyBadScheduler, verbose, message=nothing)
    # Reset counters
    s.iterations_oscillating = 0
    s.iterations_without_consistent_progress = 0
    s.iterations_without_global_progress = 0
    s.iterations_without_local_progress = 0

    # Update stepsize
    s.stepsize /= s.stepsize_change_factor
    s.n_stepsize_decreased += 1

    message = something(message, "Decreased stepsize")
    verbose && println("    $message to $(s.stepsize)")
    return s.stepsize
end

function increase_stepsize!(s::JeffreysReallyBadScheduler, verbose, message=nothing)
    # Reset counters
    s.iterations_with_slow_progress = 0
   
    # Update stepsize
    s.stepsize *= s.stepsize_change_factor
    s.n_stepsize_increased += 1

    message = something(message, "Increased stepsize")
    verbose && println("    $message to $(s.stepsize)")

    return s.stepsize
end

function stop!(s::JeffreysReallyBadScheduler, verbose, message="")
    s.stop = true
    verbose && println("    $message")
    return s.stepsize
end

############################
# Update Progress Counters #
############################

function update_progress_counters!(s::JeffreysReallyBadScheduler, error)
    @assert isfinite(error)
    @assert !s.stop
    s.error_last = s.error
    s.error = error
    s.n_iterations += 1

    # Global progress
    if s.error >= s.error_min
        s.iterations_without_global_progress += 1
        s.iterations_with_global_progress = 0
    else
        s.error_min = s.error
        s.iterations_without_global_progress = 0
        s.iterations_with_global_progress += 1
    end

    # Local progress
    local_progress_last = s.iterations_with_local_progress >= 1
    local_progress_now = s.error < s.error_last
    if local_progress_now
        s.iterations_without_local_progress = 0
        s.iterations_with_local_progress += 1
    else
        s.iterations_without_local_progress += 1
        s.iterations_with_local_progress = 0
    end

    # Slow progress
    slow_progress = -get_change_over_last(s) < s.convergence_speed_threshold
    if local_progress_now && slow_progress
        s.iterations_with_slow_progress += 1
    else
        s.iterations_with_slow_progress = 0
    end

    # Oscillation
    if xor(local_progress_last, local_progress_now)
        s.iterations_oscillating += 1
    else
        s.iterations_oscillating = 0
    end

    # Consistent progress
    if s.iterations_with_local_progress >= 4
        s.iterations_without_consistent_progress = 0
    else
        s.iterations_without_consistent_progress += 1
    end

    return s
end
