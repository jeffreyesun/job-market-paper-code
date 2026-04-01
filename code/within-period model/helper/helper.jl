
include("population_stats.jl")

# Base Overloads #
#----------------#
Base.all(p, itr1, itr2) = all(t->p(t[1],t[2]), zip(itr1, itr2))
Base.:*(s::String, i::Int) = s * string(i)
Base.:*(i::Int, s::String) = string(i) * s

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

function array_to_matrix(arr)
    s = size(arr)
    return reshape(arr, (s[1], prod(s[2:end])))
end

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


#######################
# Gumbel Distribution #
#######################

const EULER_GAMMA = 0.57721566490153286060 |> FLOAT


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
subfieldnames(T) = reduce(vcat, collect.(fieldnames.(collect(T.types))))
subfieldnamesof(::T) where T = subfieldnames(T)
fieldlabels(::T) where T = (ns = subfieldnames(T); zip(string.(ns), ns))

"Get a subfield of an object, by searching through the fields of its fields."
hasfield(T, s) = s in fieldnames(T)

fieldvalues(obj::T) where T = getfield.(Ref(obj), collect(fieldnames(T)))


####################
# Parameter Search #
####################

rms(v) = sqrt(mean(v.^2))

mysoftplus(x; a=10) = x ≥ 1/a ? x : exp(x*a-1)/a
mysoftplus_inv(y; a=10) = y ≥ 1/a ? y : (log(y*a)+1)/a
softplus_update(x, err, update_speed) = mysoftplus(mysoftplus_inv(x) + update_speed*err)


#########################
# Statefulness Reducers #
#########################

copysomething(x, y) = isnothing(x) ? y : deepcopy(x)
