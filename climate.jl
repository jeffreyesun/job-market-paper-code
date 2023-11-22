
"""
Compute climate conditions using estimated process parameters.
"""

function get_climate_path(climateparams::ClimateParams, n_dec::Int, ε_m=nothing)
    (;w0, m0, SST0, gw, β_m, ρ_SST, β_SST, σ_m) = climateparams
    if isnothing(ε_m)
        ε_m = rand(Normal(0, σ_m), n_dec)
    else
        ε_m .*= σ_m
    end

    # Initialize climate state
    logw, logm, emissions, SST = [zeros(FLOAT_PRECISION, n_dec) for _=1:4]
    logw[1] = log(w0)
    logm[1] = log(m0)
    emissions[1] = w0*m0 /1000
    SST[1] = SST0

    # Iterate climate state
    for t in 2:n_dec
        logw[t] = logw[t-1] + gw
        w = exp(logw[t-1]-15)
        logm[t] = logm[t-1] - β_m*w + ε_m[t-1]
        emissions[t] = exp(logw[t-1] + logm[t-1]) /1000

        SST[t] = ρ_SST*SST[t-1] + β_SST*emissions[t-1]
    end
    ΔSST = SST .- SST[1]

    climate_path = StructArray{ClimateState{FLOAT_PRECISION}}(;logw, logm, emissions, SST, ΔSST, ε_m)
    return pad_dims(climate_path; left=DEC_DIM-1)
end

get_ΔSST_path(SST_path::Array) = SST_path .- SST_path[1]
get_ΔSST_path(climate_path::StructArray) = get_ΔSST_path(climate_path.SST)

function get_loc_climate(loc::Location, ΔSST::Number)
    @reset loc.A = loc.A_bar * exp(ΔSST * loc.A_g)
    @reset loc.α = loc.α_bar * exp(ΔSST * loc.α_g)
    @reset loc.δ = loc.δ_bar * exp(ΔSST * loc.δ_g)
    return loc
end

function get_ΔSST_path(n_dec::Int, RCP::String, climparams::ClimateParams=ClimateParams())
    ε = Dict("RCP26"=>-0.1, "RCP45"=>0.0, "RCP60"=>0.1)[RCP]
    climpath = get_climate_path(climparams, n_dec, fill(ε, n_dec))
    return get_ΔSST_path(climpath)
end
