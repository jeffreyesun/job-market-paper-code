
##################
# Value Function #
##################

function reinterpolate!(y2::AbstractVector, y1::AbstractVector, x1::AbstractVector, x2::AbstractVector, ::Val{extrap}) where extrap
    #@assert issorted(x1) && issorted(x2)
    j = 1
    x1_j1 = x1[2]
    len_x2 = length(x2)
    len_x1m1 = length(x1) - 1

    i0 = 1

    # Possibly extrapolate to the left of x1 by clip or -Inf extrapolation.
    if extrap != :linear
        while x2[i0] < x1[1]
            @assert extrap in (:clip, -Inf)
            y2[i0] = extrap == :clip ? y1[1] : -Inf32
            i0 == len_x2 && return
            i0 += 1
        end
    end
    
    @inbounds for i=i0:len_x2
        x2_i = x2[i]
        while x2_i > x1_j1
            j == len_x1m1 && break
            j += 1
            x1_j1 = x1[j+1]
        end

        x1_j = x1[j]
        y1_j = y1[j]
        y1_j1 = y1[j+1]

        if y1_j == -Inf || y1_j1 == -Inf
            y2[i] = extrap == :clip ? max(y1_j, y1_j1) : -Inf32
        else
            slope = (y1[j + 1] - y1_j) / (x1_j1 - x1_j)
            y2[i] = slope * (x2_i - x1_j) + y1_j
        end
    end
end

function reinterpolate_arr!(y2, y1, x1, x2, ::Val{extrap}) where extrap
    @inbounds @simd for idx=CartesianIndices(tail(size(y2)))

        _tview(arr) = @view arr[:, broadcast_index(idx, tail(size(arr)))]
        reinterpolate!(_tview(y2), _tview(y1), _tview(x1), _tview(x2), Val(extrap))

    end
    return y2
end

# GPU Implementation #
#--------------------#
broadcast_index(idx, size) = CartesianIndex(ntuple(dimi -> dimi > length(size) ? 1 : min(idx[dimi], size[dimi]), Val(length(idx))))

function reinterpolate_GPU_kernel!(y2, y1, x1, x2, ::Val{extrap}) where extrap

    iv = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    Nv = prod(size(y2)[2:end])
    if iv <= Nv

        idx = CartesianIndices(tail(size(y2)))[iv]
        _tview(arr) = @view arr[:, broadcast_index(idx, tail(size(arr)))]

        @views reinterpolate!(_tview(y2), _tview(y1), _tview(x1), _tview(x2), Val(extrap))
    end
    return
end

function reinterpolate_arr!(y2::CuArray, y1::CuArray, x1::CuArray, x2::CuArray, ::Val{extrap}=Val(:linear); threads::Int = 256) where extrap
    @assert size(y1,1) == size(x1,1)
    @assert size(y2,1) == size(x2,1)

    Nv = prod(size(y2)[2:end])
    blocks = cld(Nv, threads)
    @cuda threads=threads blocks=blocks reinterpolate_GPU_kernel!(y2,y1,x1,x2,Val(extrap))
    return y2
end


######################
# State Distribution #
######################

"""
    Convert a point mass distribution y1 defined on a grid x1 to a
    pointmass distribution y2 defined on a grid x2.
    interp == :rounddown: Each point mass (x1[i], y1[i]) is assigned to the largest grid point
    x2[j] < x1[i].
    interp == :share: Each point mass (x1[i], y1[i]) is divided between the grid points
    x_new[j-1] < x1[i] <= x_new[j].
"""
function convert_distribution!(y_new, y, x, x_new, ::Val{interp}=Val(:share)) where interp
    @assert iszero(y[end])
    @assert all(>=(0), y)
    @assert issorted(x) && issorted(x_new)
    y_new .= 0

    len_x = length(x)
    len_x_new_minus_1 = length(x_new) - 1

    i0 = 1
    # Make sure that x_new[1] <= x[i]
    while x[i0] < x_new[1]
        y_new[1] += y[i0]
        i0 += 1
        i0 == len_x && return y_new
    end
    @assert x_new[1] <= x[i0]
    
    j = 1
    x_new_j_next = x_new[2]
    @inbounds for i=i0:len_x
        x_i = x[i]
        # Make sure that x_new[j] <= x[i] < x_new[j+1] <= x_new[end]
        while x_i >= x_new_j_next
            j += 1
            if j == len_x_new_minus_1
                if interp == :rounddown
                    y_new[j-1] += @views sum(y[i:end]) # What?
                elseif interp == :share
                    y_new[j] += @views sum(y[i:end])
                else
                    error("Invalid interp option $interp")
                end
                @assert sum(y_new) ≈ sum(y)
                return y_new
            end
            x_new_j_next = x_new[j+1]
        end
        @assert x_new[j] <= x_i < x_new[j+1] <= x_new[end]

        left_share = interp == :rounddown ? 1 : (x_new_j_next - x_i) / (x_new_j_next - x_new[j])
        y_new[j] += y[i] * left_share
        y_new[j+1] += y[i] * (1 - left_share)
    end
    @assert sum(y_new) ≈ sum(y)
    return y_new
end

function convert_distribution_arr!(y2, y1, x1, x2, ::Val{interp}=Val(:share)) where interp
    @inbounds @simd for idx=CartesianIndices(tail(size(y2)))

        _tview(arr) = @view arr[:, broadcast_index(idx, tail(size(arr)))]
        convert_distribution!(_tview(y2), _tview(y1), _tview(x1), _tview(x2), Val(interp))

    end
    return y2
end

# GPU Implementation #
#--------------------#
function convert_distribution_kernel!(y2, y1, x1, x2, ::Val{interp}) where interp

    iv = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    Nv = prod(size(y2)[2:end])
    if iv <= Nv

        idx = CartesianIndices(tail(size(y2)))[iv]
        _tview(arr) = @view arr[:, broadcast_index(idx, tail(size(arr)))]
        convert_distribution!(_tview(y2), _tview(y1), _tview(x1), _tview(x2), Val(interp))

    end
    return
end

function convert_distribution_arr!(y2::CuArray, y1, x1, x2, ::Val{interp}=Val(:share); threads=256) where interp
    @assert size(y1, 1) == size(x1, 1)
    @assert size(y2, 1) == size(x2, 1)

    Nv = prod(size(y2)[2:end])
    blocks = cld(Nv, threads)
    @cuda threads=threads blocks=blocks convert_distribution_kernel!(y2, y1, x1, x2, Val(interp))
    
    return y2
end


################################
# Rootfinding and Maximization #
################################

"Maximize over the sum of two vectors"
function argmax_sum(u_sub, V_sub, lb, ub)
    max = FLOAT(-Inf)
    argmax = lb

    @inbounds for i=lb:ub
        val = u_sub[i] + V_sub[i]
        if val > max
            max = val
            argmax = i
        end
    end
    return argmax
end

"""
    For each pre-utility state, maximize utility + post-utility value,
    over the post-utility states.
"""
function k1_argmax!(V_prec, k1, V, u)
    n = length(k1)
    @assert ispow2(n-1)
    k1[1] = 1
    V_prec[1] = u[1,1] + V[1]

    k1[end] = @views argmax_sum(u[:,end], V, 1, n)
    V_prec[end] = u[k1[end], end] + V[k1[end]]
    
    segment_length = div(n-1, 2)
    while segment_length >= 1
        i = 1
        while i < n - 1
            k1_lb = k1[i]
            i += segment_length
            k1_ub = k1[i+segment_length]
            
            k1[i] = @views argmax_sum(u[:,i], V, k1_lb, min(k1_ub,i))
            V_prec[i] = u[k1[i], i] + V[k1[i]]
            
            i += segment_length
        end
        segment_length = div(segment_length, 2)
        #TODO figure out how to do for non-power-of-two vector lengths
    end
    return k1
end

"Array version of k1_argmax!, operating along first dimension."
function k1_argmax_arr!(V_prec, k1, V, LOG_EXPENDITURE_MAT)
    @assert size(V_prec) == size(k1) == size(V)
    mat_size = (size(V_prec, 1), prod(size(V_prec)[2:end]))
    V_prec_mat = reshape(V_prec, mat_size)
    k1_mat = reshape(k1, mat_size)
    V_mat = reshape(V, mat_size)

    @inbounds @simd for coli in 1:mat_size[2]
        @views k1_argmax!(V_prec_mat[:,coli], k1_mat[:, coli], V_mat[:, coli], LOG_EXPENDITURE_MAT)
    end
    return k1
end

"Slow array version of k1_argmax_arr!, for verification and testing."
function k1_argmax_arr_slow!(V_prec, k1, V, LOG_EXPENDITURE_MAT)
    @assert size(V_prec) == size(k1) == size(V)
    mat_size = (size(V_prec, 1), prod(size(V_prec)[2:end]))
    V_prec_mat = reshape(V_prec, mat_size)
    k1_mat = reshape(k1, mat_size)
    V_mat = reshape(V, mat_size)

    for iv in 1:mat_size[2]
        u_plus_V = V_mat[:, iv]' .+ LOG_EXPENDITURE_MAT'
        k1_mat[:, iv] .= [argmax(u_plus_V[rowi,:]) for rowi in 1:N_K]
        V_prec_mat[:, iv] .= maximum(u_plus_V; dims=2)
    end
    return replace!(k1, 129=>1)
end

# GPU Implementation #
#--------------------#
function k1_argmax_kernel!(V_prec, k1, V, u)
    iv = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    Nv = size(V_prec, 2)
    if iv <= Nv
        n = size(k1, 1)
        @assert ispow2(n-1)
        k1[1,iv] = 1
        V_prec[1,iv] = u[1,1] + V[1,iv]

        # k1[end,iv] = argmax_sum_GPU(u[:,end], V[:,iv], 1, n)
        ## argmax_sum_GPU begin
        max = FLOAT(-Inf)
        argmax = 1

        @inbounds for j=1:n
            val = u[j,end] + V[j,iv]
            if val > max
                max = val
                argmax = j
            end
        end

        k1[end,iv] = argmax
        ## argmax_sum_GPU end

        V_prec[end,iv] = u[k1[end,iv], end] + V[k1[end,iv],iv]
        
        segment_length = div(n-1, 2)
        while segment_length >= 1
            i = 1
            while i < n - 1
                k1_lb = k1[i,iv]
                i += segment_length
                k1_ub = k1[i+segment_length,iv]
                
                # k1[i,iv] = argmax_sum_GPU(u[:,i], V[:,iv], k1_lb, min(k1_ub,i))
                ## argmax_sum_GPU begin
                max = FLOAT(-Inf)
                argmax = k1_lb

                @inbounds for j=k1_lb:min(k1_ub,i)
                    val = u[j,i] + V[j,iv]
                    if val > max
                        max = val
                        argmax = j
                    end
                end
                k1[i,iv] = argmax
                ## argmax_sum_GPU end

                V_prec[i,iv] = u[k1[i,iv], i] + V[k1[i,iv],iv]
                
                i += segment_length
            end
            segment_length = div(segment_length, 2)
            #TODO figure out how to do for non-power-of-two vector lengths
        end
    end
    return
end

function k1_argmax_arr!(V_prec::CuArray, k1, V, u; threads::Int = 256)
    @assert size(V_prec) == size(k1) == size(V)
    mat_size = (size(V_prec, 1), prod(size(V_prec)[2:end]))
    V_prec_mat = reshape(V_prec, mat_size)
    k1_mat = reshape(k1, mat_size)
    V_mat = reshape(V, mat_size)
    
    Nv = size(V_prec_mat, 2)
    blocks = cld(Nv, threads)
    @cuda threads=threads blocks=blocks k1_argmax_kernel!(V_prec_mat, k1_mat, V_mat, u)
    return k1
end
