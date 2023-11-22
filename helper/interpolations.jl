
##################
# Value Function #
##################

function reinterpolate!(y2::AbstractVector, y1::AbstractVector, x1::AbstractVector, x2::AbstractVector, left_extrap::Val{:linear}=Val(:linear))
    #@assert issorted(x1) && issorted(x2)
    j = 1
    x1_j1 = x1[2]
    len_x1m1 = length(x1) - 1
    
    #@inbounds
    for i=1:length(x2)
        x2_i = x2[i]
        while x2_i > x1_j1
            if j == len_x1m1
                break
            end
            j += 1
            x1_j1 = x1[j+1]
        end

        x1_j = x1[j]
        y1_j = y1[j]
        y1_j1 = y1[j+1]

        if y1_j == -Inf || y1_j1 == -Inf
            y2[i] = -Inf
        else
            slope = (y1[j + 1] - y1_j) / (x1_j1 - x1_j)
            y2[i] = slope * (x2_i - x1_j) + y1_j
        end
    end
end

function reinterpolate!(y2::AbstractVector, y1::AbstractVector, x1::AbstractVector, x2::AbstractVector, left_extrap::Val{-Inf})
    #@assert issorted(x1) && issorted(x2)
    j = 1
    x1_j1 = x1[2]
    len_x2 = length(x2)

    i0 = 1

    while x2[i0] < x1[1]
        y2[i0] = -Inf32
        if i0 == len_x2
            return
        else
            i0 += 1
        end
    end
    
    #@inbounds
    for i=i0:length(x2)
        x2_i = x2[i]
        while x2_i > x1_j1
            if j == length(x1) - 1
                break
            end
            j += 1
            x1_j1 = x1[j+1]
        end

        x1_j = x1[j]
        y1_j = y1[j]
        y1_j1 = y1[j+1]

        if y1_j == -Inf || y1_j1 == -Inf
            y2[i] = -Inf32
        else
            slope = (y1[j + 1] - y1_j) / (x1_j1 - x1_j)
            y2[i] = slope * (x2_i - x1_j) + y1_j
        end
    end
    return
end

function reinterpolate!(y2::AbstractVector, y1::AbstractVector, x1::AbstractVector, x2::AbstractVector, left_extrap::Val{:clip})
    #return reinterpolate!(y2, y1, x1, x2, Val(y1[1]))
    #@assert issorted(x1) && issorted(x2)
    j = 1
    x1_j1 = x1[2]
    len_x2 = length(x2)

    i0 = 1

    while x2[i0] < x1[1]
        y2[i0] = y1[1]
        if i0 == len_x2
            return
        else
            i0 += 1
        end
    end
    
    #@inbounds
    for i=i0:length(x2)
        x2_i = x2[i]
        while x2_i > x1_j1
            if j == length(x1) - 1
                break
            end
            j += 1
            x1_j1 = x1[j+1]
        end

        x1_j = x1[j]
        y1_j = y1[j]
        y1_j1 = y1[j+1]

        if y1_j == -Inf || y1_j1 == -Inf
            y2[i] = y1[1]
        else
            slope = (y1[j + 1] - y1_j) / (x1_j1 - x1_j)
            y2[i] = slope * (x2_i - x1_j) + y1_j
        end
    end
end


######################
# State Distribution #
######################

"""
    Convert a point mass distribution y1 defined on a grid x1 to a
    pointmass distribution y2 defined on a grid x2.
    Each point mass (x1[i], y1[i]) is assigned to the largest grid point
    x2[j] < x1[i].
"""
function convert_distribution_pointmass_rounddown!(y_new, y, x, x_new)
    @assert iszero(y[end])
    @assert all(>=(0), y)
    #@assert issorted(x) && issorted(x_new)
    y_new .= 0
    len_x_new_minus_1 = length(x_new) - 1

    i = 1
    # Make sure that x_new[1] <= x[i]
    while x[i] < x_new[1]
        y_new[1] += y[i]
        i += 1
    end
    @assert x_new[1] < x[i]
    
    j = 1
    x_new_j_next = x_new[2]
    @inbounds for i=i:length(x)
        x_i = x[i]
        # Make sure that x_new[j] <= x[i] < x_new[j+1] <= x_new[end]
        while x_i >= x_new_j_next
            j += 1
            if j == len_x_new_minus_1
                y_new[j-1] += @views sum(y[i:end])
                @assert sum(y_new) ≈ sum(y)
                return y_new
            end
            x_new_j_next = x_new[j+1]
        end
        @assert x_new[j] <= x_i < x_new[j+1] <= x_new[end]

        y_new[j] += y[i]
    end
    @assert sum(y_new) ≈ sum(y)
    return y_new
end

"""
    Convert a point mass distribution y defined on a grid x to a
    pointmass distribution y_new defined on a grid x_new.
    Each point mass (x[i], y[i]) is divided between the grid points
    x_new[j-1] < x[i] <= x_new[j].
"""
function convert_distribution_pointmass_share!(y_new, y, x, x_new)
    @assert iszero(y[end])
    @assert all(>=(0), y)
    @assert issorted(x) && issorted(x_new)
    y_new .= 0
    
    len_x = length(x)
    len_x_new_minus_1 = length(x_new) - 1

    @inline validate(y_new) = y_new
    # validate(y_new) = validate_converted_distribution(y_new, y)
    
    i = 1
    # Make sure that x_new[1] <= x[i]
    while x[i] < x_new[1]
        y_new[1] += y[i]
        i += 1
        if i == len_x
            #@assert sum(y_new) ≈ sum(y)
            return validate(y_new)
        end
    end
    #@assert x_new[1] <= x[i]
    
    j = 1
    x_new_j_next = x_new[2]
    @inbounds for i=i:len_x
        x_i = x[i]
        # Make sure that x_new[j] <= x[i] < x_new[j+1] <= x_new[end-1]
        while x_i >= x_new_j_next
            j += 1
            if j == len_x_new_minus_1
                y_new[j] += @views sum(y[i:end])
                return validate(y_new)
            end
            x_new_j_next = x_new[j+1]
        end
        #@assert x_new[j] <= x_i < x_new[j+1] <= x_new[end-1]

        left_share = (x_new_j_next - x_i) / (x_new_j_next - x_new[j])
        y_new[j] += y[i] * left_share
        y_new[j+1] += y[i] * (1 - left_share)
    end

    return validate(y_new)
end

"""
    Convert a piecewise uniform distribution y1 defined on a grid x1 to a
    piecewise uniform distribution y2 defined on a grid x2. This is an
    interpolation, not exact.
"""
function convert_distribution_piecewiseuniform!(y2, y1, x1, x2)
    @assert iszero(y1[end])
    @assert all(>=(0), y1)
    #@assert issorted(x1) && issorted(x2)

    y2 .= 0

    len_x1 = length(x1)
    mass_next = 0
    reached_end = false
    x1_l_ind = 1
    x1_r = x1[2]

    # Allow x1 to start before x2, but only if the distribution is 0 there
    while x1[x1_l_ind] < x2[1]
        @assert y1[x1_l_ind] == 0
        if x1_l_ind == len_x1 - 1
            return
        else
            x1_l_ind += 1
        end
    end
    x1_r = x1[x1_l_ind + 1]
    @assert x2[1] <= x1[x1_l_ind]
    
    x2_r_ind = 2
    while x2_r_ind <= length(x2) && x2[x2_r_ind] < x1[x1_l_ind]
        x2_r_ind += 1
    end
    
    x_pos = x1[x1_l_ind]
    while (!reached_end) & (x2_r_ind < length(x2))
        while (!reached_end) & (x1_r < x2[x2_r_ind])
            mass_share = x_pos == x1[x1_l_ind] ? 1 : (x1_r - x_pos) / (x1_r - x1[x1_l_ind])
            mass_next += y1[x1_l_ind] * mass_share

            x_pos = x1_r

            x1_l_ind += 1

            reached_end = x1_l_ind == len_x1

            x1_r = reached_end ? Inf : x1[x1_l_ind + 1]
        end

        mass_share = (x2[x2_r_ind] - x_pos) / (x1_r - x1[x1_l_ind])
        mass_next += y1[x1_l_ind] * mass_share
        y2[x2_r_ind-1] += mass_next

        mass_next = 0
        x_pos = x2[x2_r_ind]
        x2_r_ind += 1
    end

    if !reached_end
        mass_next = y1[x1_l_ind] * (x1_r - x_pos) / (x1_r - x1[x1_l_ind])
        for j in (x1_l_ind+1):len_x1
            mass_next += y1[j]
        end
        y2[end-1] += mass_next
    end

    return y2
end

const convert_distribution! = convert_distribution_pointmass_share!
