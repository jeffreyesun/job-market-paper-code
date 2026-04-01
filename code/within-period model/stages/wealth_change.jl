
#########################
# General Wealth Change #
#########################

# Backward #
#----------#
"Reinterpolate a value function from after a wealth change to before."
function apply_wealth_change_V!(V_pre, V_post, wealth_post, left_extrap=Val(:linear))
    wealth_grid = V_pre isa CuArray ? WEALTH_GRID_FLAT_GPU : WEALTH_GRID_FLAT
    return reinterpolate_arr!(V_pre, V_post, wealth_grid, wealth_post, left_extrap)
end

# Forward #
#---------#
function apply_wealth_change_λ!(λ_post, λ_pre, wealth_post)
    wealth_grid = λ_pre isa CuArray ? WEALTH_GRID_FLAT_GPU : WEALTH_GRID_FLAT
    return convert_distribution_arr!(λ_post, λ_pre, wealth_post, wealth_grid)
end
