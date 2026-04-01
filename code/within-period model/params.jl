

"""Units
Money: Thousand 2020 USD
Housing: Bedroom-equivalent
Time/Age: Decade
Population: 1000s of people
Location: 2020 PUMA
"""

##############
# Parameters #
##############

# Standard Parameters #
#---------------------#
const β = 0.98^10 |> FLOAT
#const r = 1/β - 1 |> FLOAT
const r = 0.254 |> FLOAT
#const agglom_force = 1.1

# Data, Non-Spatial #
#-------------------#
const LOGZ_AGE = FLOAT[6.241, 6.961, 6.970, 6.996, 6.785, 6.507] # 2022 SCF
const WEALTH_MAX = 1e8 / 1e3 |> FLOAT
const Z_PERSISTENCE = 0.737 |> FLOAT # Decadalized KMV
const Z_STD = 0.56 |> FLOAT # Decadalized KMV

const POP_SUM = sum(read_location_moments().pop)


###############
# State Space #
###############

# State Space Grids #
#-------------------#
if N_Z > 1
    const Z_PROCESS = rouwenhorst(N_Z, Float64(Z_PERSISTENCE), Float64(Z_STD), 0)
else
    const Z_PROCESS = (;p = Float32[1;;], state_values = Float32[0])
end
const WEALTH_GRID_FLAT = FLOAT.(exp.(range(0, stop=log(WEALTH_MAX./10), length=N_K)).*10 .- 10)
const LOGZ_GRID_FLAT = FLOAT.(Z_PROCESS.state_values)
const H_GRID_FLAT = FLOAT[0, 0.3, 0.7, (1:4).^2...]
const βZ_T = FLOAT.(β.*Z_PROCESS.p .+ 1e-15)
const Z_T_TRANSPOSE = FLOAT.(Z_PROCESS.p' .+ 1e-15)

# Shaped Grids for Broadcasting #
#-------------------------------#
const WEALTH_GRID = pad_dims(WEALTH_GRID_FLAT; ndims_new=4)
const WEALTH_NEXT_GRID = pad_dims(WEALTH_GRID_FLAT; left=5)
const LOGZ_GRID = pad_dims(LOGZ_GRID_FLAT; left=1, ndims_new=4)
const Z_GRID = [exp.(LOGZ_GRID .+ LOGZ_AGE[agei]) for agei=1:N_AGE]
const H_GRID = pad_dims(H_GRID_FLAT; left=2, ndims_new=4)
const AGE_GRID = pad_dims(1:N_AGE; left=4, ndims_new=5)

const LOG_EXPENDITURE_MAT = @. FLOAT(log(max(WEALTH_GRID_FLAT' - WEALTH_GRID_FLAT, 1e-2)))
