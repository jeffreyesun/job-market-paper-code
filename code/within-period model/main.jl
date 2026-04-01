
using LinearAlgebra
using Statistics
using StructArrays
using JLD2
using DataFrames
using CSV
using GLM
using CUDA
using Printf
import Base: tail
import Base.Threads: @threads
import Distributions: Normal, LogNormal, pdf
import QuantEcon: rouwenhorst
import Random: seed!
import CUDA: allowscalar


#############
# Constants #
#############

const FLOAT = Float32
const BASELINE_YEAR = 2020
const ACTIVELY_DISPATCH_TO_GPU = false

###############
# State Space #
###############

# Grid sizes
const N_K = 129
const N_Z = 5
const N_H = 7
const N_LOC = 100
const N_AGE = 6
const STATE_IDXs = (N_K, N_Z, N_H, N_LOC)
const STATE_IDXs_FULL = (STATE_IDXs..., N_AGE)
# State space dimension layout
const K_DIM, Z_DIM, H_DIM, LOC_DIM, AGE_DIM, K_NEXT_DIM, DEC_DIM = 1:7
# CartesianIndex bounds
const RENT_BOUND = CartesianIndex((N_K, N_Z, 1, N_LOC))

##############
# Load Model #
##############

# Climate Module #
include("climate.jl")
using .ClimateProcess: N_SCENARIO, SCENARIO_IND_INIT, SST_TRANSMAT, SST_GRID, draw_scenario_ind, draw_scenario_path

# Economic Module # (not yet actually modularized)
include("types.jl")
include("helper/helper.jl")
include("read_data.jl")
include("params.jl")
include("initialize_globals.jl")
set_globals_from_data()
include("preallocate.jl")
include("types_gpu.jl")
include("precompute.jl")
include("stages/stages_main.jl")
include("household_problem.jl")
include("entry.jl")
include("simulated_moments.jl")
include("housing_supply.jl")
include("within_period_stage_solution.jl")
