
using LinearAlgebra
using Statistics
using Accessors
using StructArrays
using JLD2
import Base: tail
import Base.Threads: @threads

###############
# State Space #
###############

# Grid sizetarget_moments = get_empirical_aggregate_moments()
const N_K = 65
const N_Z = 5
const N_Hli = 7
const N_Hle = 5
const N_LOC = 1713
const N_AGE = 6
const STATE_IDXs = (N_K, N_Z, N_Hli, N_Hle, N_LOC)
const STATE_IDXs_FULL = (STATE_IDXs..., N_AGE)
# State space dimension layout
const K_DIM, Z_DIM, H_LI_DIM, H_LE_DIM, LOC_DIM, AGE_DIM, K_NEXT_DIM, DEC_DIM = 1:8

##############
# Load Model #
##############

include("types.jl")
include("helper.jl")
include("read_data.jl")
include("params.jl")
include("preallocate.jl")
include("model.jl")
include("transition_functions.jl")
include("simulate.jl")
include("steady_state.jl")
include("climate.jl")
include("transition_dynamics.jl")
include("global_solution.jl")
include("validate.jl")
