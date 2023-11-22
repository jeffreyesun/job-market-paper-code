
include("helper/parameter_scheduling.jl")
include("helper/interpolations.jl")

# Indexing #
#----------#

function transposedims(v, transpositions, inplace=true)
    perm = get_perm(ndims(v), transpositions)

    if inplace
        return PermutedDimsArray(v, perm)
    else
        return permutedims(v, perm)
    end
end

function transposedims(v, transpositions::T, inplace=true) where T <: Tuple{<:Number, <:Number}
    return transposedims(v, [transpositions], inplace)
end

function get_perm(n, transpositions)
    perm = collect(1:n)
    for t in transpositions
        perm[[t[1], t[2]]] .= perm[t[2]], perm[t[1]]
    end
    return perm
end

function light_argmax(x, dims)
    return argmax.(eachslice(x, dims=dimsexcept(x, dims), drop=false))
end

function getindex_slices(x, dims, i)
    return getindex.(eachslice(x, dims=dimsexcept(x, dims), drop=false), i)
end


"""
Like selectdim, but follow broadcasting rules. If the size of the
dimension is 1, return the whole array.
"""
function selectdim_broadcasty(A::AbstractArray, d::Integer, i::Int)
    size(A, d) == 1 && return A
    return selectdim(A, d, i)
end


# https://stackoverflow.com/a/69470628/4828492
function extend_dims(A, which_dim, n=1)
    s = [size(A)...]
    for _ in 1:n
        insert!(s,which_dim,1)
    end
    return reshape(A, s...)
end

function extend_dims(A, dims::NTuple{N,T}) where {N, T <: Tuple}
    return reduce((A, dn)->extend_dims(A, dn...), dims, init=A)
end

function pad_dims(A, left, right)
    s = Int[ones(left)..., size(A)..., ones(right)...]
    return reshape(A, s...)
end

function pad_dims(A; left=0, right=0, ndims_new=-1)
    if ndims_new != -1
        if right == 0
            right = ndims_new - ndims(A) - left
        else
            left = ndims_new - ndims(A) - right
        end
    end
    return pad_dims(A, left, right)
end

e_vec(n, i) = [i==j for j=1:n]

# Slicing #
#---------#

struct DimsExcept{T}
    dims::T
end

function dimsexcept(A::AbstractArray{T, N}, dims::NTuple{M}) where {T,N,M}
    return NTuple{N-M,Int}(i for i=1:N if i ∉ dims)
end
dimsexcept(A, dim::Int) = dimsexcept(A, (dim,))
dimsexcept(A, dims::DimsExcept) = dimsexcept(A, dims.dims)

#= in jail until they can be made to work fast=#
Base.eachslice(A, dims, drop=true) = eachslice(A; dims, drop)
Base.eachslice(A, dims::DimsExcept, drop=true) = eachslice(A; dims=dimsexcept(A, dims), drop)

_eachslice_catchRef(Ai; dims, drop) = eachslice(Ai; dims, drop)
_eachslice_catchRef(Ai::Ref; dims, drop) = Ai

function broadcastslices(f, dims, A...; args=[])
    return f.(eachslice.(A; dims, drop=false)..., Ref.(args)...)
end

function broadcastslices(f, dims::DimsExcept, A...; args=[])
    return broadcastslices(f, dimsexcept(A[1], dims), A...; args)
end
#=
function eachslice_asview_along(A::AbstractArray, dims)
    nd = ndims(A)
    longdims = ntuple(d -> d in dims ? 1 : size(A,d), nd)
    sliced = .!in.(1:nd, Ref(dims))
    #[view(A, ntuple(d -> sliced[d] ? Colon() : (idx[d]:idx[d]), nd)...) for idx in CartesianIndices(longdims)]
    return (view(A, ntuple(d -> (idx[d] : (sliced[d] ? size(A,d) : idx[d])), nd)...) for idx in CartesianIndices(longdims))
end

eachslice_asview(A::AbstractArray, dims) = eachslice_asview_along(A, dimsexcept(A, dims))
=#


# Rootfinding and Maximization #
#------------------------------#

function isquasiconvex(v)
    v_last = v[1]
    increasing = v[2] >= v[1]
    for v_i in v[2:end]
        if v_i > v_last
            if !increasing
                return false
            end
        elseif v_i < v_last
            increasing = false
        end
        v_last = v_i
    end
    return true
end

function _get_right_start(v::AbstractVector)
    right = length(v)
    while right > 1 && v[right] == -Inf 
        right = div(right, 2)
    end
    return right*2
end

"Find the argmax of an array iff the array is quasiconvex"
function quasiconvex_argmax(v; right::Int=length(v))    
    left = 1
    
    while right - left > 1
        gap = div(right - left, 3)
        middle_l = left + gap
        middle_r = right - gap

        @inbounds if v[middle_l] < v[middle_r]
            left = middle_l + 1
        else
            right = middle_r - 1
        end
    end

    return v[left] > v[right] ? left : right
end

"""
    For each pre-utility state, maximize utility + post-utility value,
    over the post-utility states.
"""
function k1_argmax!(V_prec, k1, V, u)
    n = length(k1)
    @assert ispow2(n-1)
    k1[1] = 1
    V_prec[1] = u[1][1] + V[1]

    k1[end] = argmax_sum(u[end], V, 1, n)
    V_prec[end] = u[end][k1[end]] + V[k1[end]]
    
    segment_length = div(n-1, 2)
    while segment_length >= 1
        i = 1
        while i < n - 1
            k1_lb = k1[i]
            i += segment_length
            k1_ub = k1[i+segment_length]
            
            k1[i] = argmax_sum(u[i], V, k1_lb, min(k1_ub,i))
            V_prec[i] = u[i][k1[i]] + V[k1[i]]
            
            i += segment_length
        end
        segment_length = div(segment_length, 2)
        #TODO figure out how to do for non-power-of-two vector lengths
    end
    return k1
end

"Maximize over the sum of two vectors"
function argmax_sum(u_sub, V_sub, lb, ub)
    max = -Inf
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


#######################
# Gumbel Distribution #
#######################

const EULER_GAMMA = 0.57721566490153286060 |> FLOAT_PRECISION

##############################
# Exponential Approximations #
##############################

const fastexp = Base.Math.exp_fast
const fastlog = log

##########
# Macros #
##########

macro print(ex)
    return :(println($(esc(ex))))
end

##########
# Fields #
##########

"Get all fields of the components of a struct."
subfieldnames(::T) where T = reduce(vcat, collect.(fieldnames.(collect(T.types))))

"Get a subfield of an object, by searching through the fields of its fields."
hasfield(T, s) = s in fieldnames(T)

fieldvalues(obj::T) where T = getfield.(Ref(obj), collect(fieldnames(T)))
