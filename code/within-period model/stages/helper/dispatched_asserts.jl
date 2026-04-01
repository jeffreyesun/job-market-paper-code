
assert_nonan(arr) = @assert !any(isnan, arr)
assert_nonan(::CuArray) = nothing

assert_all_finite(arr) = @assert all(isfinite, arr)
assert_all_finite(::CuArray) = nothing

assert_sum_approx(arr1, arr2) = @assert sum(arr1) ≈ sum(arr2)
assert_sum_approx(::CuArray, ::CuArray) = nothing

assert_equal(arr1, arr2) = @assert arr1 == arr2
assert_equal(::CuArray, ::CuArray) = nothing

assert_approx(arr1, arr2) = @assert arr1 ≈ arr2
assert_approx(::CuArray, ::CuArray) = nothing
