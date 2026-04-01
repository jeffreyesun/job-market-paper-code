
########################
# Borrowing Constraint #
########################

# Backward #
#----------#
"""
    Convert any values below the borrowing constraint to -Inf.
    Must be applied whenever the household makes a decision that affects
    the borrowing constraint. I.e. saving or buying a house.
"""
enforce_borrowing_constraint!(V, prealloc) = V .+= prealloc.forbidden_states

assert_renter_V_finite(V) = @assert all(isfinite, V[2:end,:,1,:]) # Renters with money should always have finite value
assert_renter_V_finite(V::CuArray) = nothing
