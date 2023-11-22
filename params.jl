
import Distributions: Normal, LogNormal, pdf
import QuantEcon: tauchen
import Random: seed!

##############
# Parameters #
##############

# Standard Parameters #
#---------------------#
const β = 0.98^10 |> FLOAT_PRECISION
const r = 1/β - 1 |> FLOAT_PRECISION
#const agglom_force = 1.1

# Data, Non-Spatial #
#-------------------#
const LOGZ_AGE = FLOAT_PRECISION.(log(10) .+ [10.806, 11.215, 11.338, 11.263, 10.985, 10.821] .- log(1000))
const WEALTH_MAX = 1e7 / 1e3 |> FLOAT_PRECISION
const Z_PERSISTENCE = 0.9 |> FLOAT_PRECISION
const Z_STD = 0.2 |> FLOAT_PRECISION

# Data, Spatial #
#---------------#

const DIST_LOC = FLOAT_PRECISION.(read_pairwise_distances())


###############
# State Space #
###############

# State Space Grids #
#-------------------#
if N_Z > 1
    const Z_PROCESS = tauchen(N_Z, Z_PERSISTENCE, Z_STD, 0)
else
    const Z_PROCESS = (;p = Float32[1;;], state_values = Float32[0])
end
const WEALTH_GRID_FLAT = FLOAT_PRECISION.(exp.(range(0, stop=log(WEALTH_MAX./10), length=N_K)).*10 .- 10)
const LOGZ_GRID_FLAT = FLOAT_PRECISION.(Z_PROCESS.state_values)
const H_LIVE_GRID_FLAT = FLOAT_PRECISION[0, 0.7, 1, 2, 4, 8, 16]
const H_LET_GRID_FLAT = FLOAT_PRECISION[0, 8, 16, 64, 128]
const βZ_T = FLOAT_PRECISION.(β.*Z_PROCESS.p .+ 1e-15)
const Z_T_TRANSPOSE = FLOAT_PRECISION.(Z_PROCESS.p' .+ 1e-15)

# Shaped Grids for Broadcasting #
#-------------------------------#
const WEALTH_GRID = pad_dims(WEALTH_GRID_FLAT; ndims_new=5)
const WEALTH_NEXT_GRID = pad_dims(WEALTH_GRID_FLAT; left=6)
const LOGZ_GRID = pad_dims(LOGZ_GRID_FLAT; left=1, ndims_new=5)
const Z_GRID = [exp.(LOGZ_GRID .+ LOGZ_AGE[agei]) for agei=1:N_AGE]
const H_LIVE_GRID = pad_dims(H_LIVE_GRID_FLAT; left=2, ndims_new=5)
const H_LET_GRID = pad_dims(H_LET_GRID_FLAT; left=3, ndims_new=5)
const AGE_GRID = pad_dims(1:N_AGE; left=5, ndims_new=6)

const X_GRID = [(WEALTH_GRID, Z_GRID[agei], H_LIVE_GRID, H_LET_GRID) for agei=1:N_AGE]


########################
# Initial Distribution #
########################

const YOUNG_WEALTH_INIT = get_wealth_grid_init(read_empirical_target_moments()[2])
# Normal pdf sampled at state space grid points
const YOUNG_Z_FLAT = FLOAT_PRECISION.(normalize(pdf.(Normal(), LOGZ_GRID_FLAT), 1))
