

#################################
# Compute Population Statistics #
#################################

weightedmean(vals, weights; dims=:) = sum(vals .* weights; dims) ./ sum(weights; dims)

maybesumoverages(dims, means, weightvec) = (dims==(:)||AGE_DIM in dims) ? weightedmean(means, sum.(weightvec)) : means

function weightedmean(valvec::Vector{<:Array}, weightvec::Vector{<:Array}; dims=:)
    means = [weightedmean(val, weight; dims) for (val, weight) in zip(valvec, weightvec)]
    return maybesumoverages(dims, means, sum.(weightvec))
end

function weightedmean(vals, weightvec::Vector{<:Array}; dims=:)
    means = [weightedmean(vals, weight; dims) for weight in weightvec]
    return maybesumoverages(dims, means, sum.(weightvec))
end

function weightedmean(valvec::Vector{<:Array}, weights; dims=:)
    means = [weightedmean(val, weights; dims) for val in valvec]
    return maybesumoverages(dims, means, ones(N_AGE))
end
