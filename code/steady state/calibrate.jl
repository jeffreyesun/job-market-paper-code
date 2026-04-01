
#############################
# Calibrate Spatial α and A #
############################# 

# Inner Function #
#----------------#
function get_pops_and_earnings_from_αAρq!(pd, params, α, A, ρ, q; H_S=nothing)
    params.α_bar .= α
    params.A_bar .= A
    pd.prices.ρ .= ρ
    pd.prices.q .= q
    pd = solve_household_problem_steady_state!(pd, params; H_S)

    pops = get_pops(pd)
    mean_earnings_workage = get_mean_earnings(pd, params; ages=1:4)
    H_rent = get_H_rent(pd, params)
    H_D = get_H_D(pd)
    return (;pops, mean_earnings_workage, H_rent, H_D)
end

# Calibration #
#-------------#
function _calibrate_spatial!(params, pd, data_moments=read_location_moments(); update_speed_base=0.1, rtol=1e-4, verbosity=0, save=false, save_intermediate=false, fn="save.csv")
    (;prices) = pd

    # Data Moments
    data_pops = data_moments.pop ./ mean(data_moments.pop)
    data_earnings_workage = data_moments.mean_earn_workage ./ mean(data_moments.mean_earn_workage)
    H_S = data_moments.H
    # Correct for empirical relationship between square footage and mean earnings
    # Texas NYC: log(2400/1578)/log(53/128) = -0.47
    H_S .*= (data_earnings_workage/mean(data_earnings_workage)).^(-0.47)
    
    # Initial Guesses
    α_guess = copy(params.α)
    A_guess = copy(params.A)
    ρ_guess = copy(prices.ρ)
    q_guess = copy(prices.q)

    # Instantiate Model Data
    pd = solve_household_problem_steady_state!(pd, params)

    # Spatial Calibration Loop
    spatial_err = Inf
    n = 0
    while spatial_err > rtol
        (;pops, mean_earnings_workage, H_rent, H_D) = get_pops_and_earnings_from_αAρq!(pd, params, α_guess, A_guess, ρ_guess, q_guess; H_S)
        params.Π .= H_S.*q_guess.^-params.elasticity
        pops ./= mean(pops)
        mean_earnings_workage ./= mean(mean_earnings_workage)

        pop_err = @. (pops - data_pops)/data_pops
        income_err = @. (mean_earnings_workage - data_earnings_workage)/data_earnings_workage
        H_rent_err = @. (H_rent - H_D)/H_D
        H_D_err = @. (H_D - H_S)/H_S

        update_speed = update_speed_base/10#*cos(n/20)^2
        n <= 5 && (update_speed *= 0.1)

        @. α_guess .= softplus_update(α_guess, -pop_err, update_speed)
        @. A_guess .= softplus_update(A_guess, -income_err, update_speed*2)
        @. ρ_guess .= softplus_update(ρ_guess, H_rent_err, update_speed*50)
        #@. q_guess .= softplus_update(q_guess, H_rent_err*q_guess/ρ_guess*0.5, update_speed*100)
        @. q_guess .= softplus_update(q_guess, H_D_err, update_speed*100)
        @assert all(>(1e-3), q_guess)

        α_guess ./= mean(α_guess)
        A_guess ./= mean(A_guess)

        spatial_err = rms(pop_err) + rms(income_err) + rms(H_rent_err) + rms(H_D_err)

        if verbosity >= 1
            @show spatial_err
            @show rms(pop_err)
            @show rms(income_err)
            @show rms(H_rent_err)
            @show rms(H_D_err)
        end
        n += 1

        if save_intermediate
            if pd isa PeriodDataGPU
                params_write = to_cpu(params)
                prices_write = to_cpu(prices)
            else
                params_write = params
                prices_write = prices
            end
            CSV.write(INTERMEDIATE_DATA_3_DIR*"$(N_LOC)loc_temp.csv", DataFrame(;
                PUMA2020 = params_write.GISMATCH[:],
                α_bar = params_write.α_bar[:],
                A_bar = params_write.A_bar[:],
                ρ = prices_write.ρ[:],
                q = prices_write.q[:],
                Π = params_write.Π[:],
                H_S = params_write.H_S[:],
            ))
        end
    end

    if save
        if pd isa PeriodDataGPU
            params_write = to_cpu(params)
            prices_write = to_cpu(prices)
        else
            params_write = params
            prices_write = prices
        end
        CSV.write(INTERMEDIATE_DATA_3_DIR*"$(N_LOC)loc.csv", DataFrame(;
            PUMA2020 = params_write.GISMATCH[:],
            α_bar = params_write.α_bar[:],
            A_bar = params_write.A_bar[:],
            ρ = prices_write.ρ[:],
            q = prices_write.q[:],
            Π = params_write.Π[:],
            H_S = params_write.H_S[:],
        ))
    end
    
    return (;pd, params, prices)
end

# Wrap according to value of ACTIVELY_DISPATCH_TO_GPU #
#-----------------------------------------------------#
function calibrate_spatial!(params, pd; kwargs...)

    # Run on GPU if ACTIVELY_DISPATCH_TO_GPU == true
    if ACTIVELY_DISPATCH_TO_GPU && pd isa PeriodData
        # Move inputs to GPU
        pd_g = to_gpu(pd)
        params_g = to_gpu(params)
        
        # Solve on GPU
        data_moments = to_gpu(read_location_moments())
        pd_params_g = _calibrate_spatial!(params_g, pd_g, data_moments; kwargs...)
        
        # Move output back to CPU
        pd = to_cpu!(pd, pd_params_g.pd)
        params = to_cpu!(params, pd_params_g.params)
    else
        data_moments = pd isa PeriodData ? read_location_moments() : to_gpu(read_location_moments())
        (;pd, params) = _calibrate_spatial!(params, pd, data_moments; kwargs...)
    end
    return (;pd, params, prices=pd.prices)
end

# Load Saved Parameters #
#-----------------------#
"Load partially-calibrated spatial parameters"
function load_temp_params!(pd, params)
    saved_params = CSV.read(INTERMEDIATE_DATA_3_DIR*"$(N_LOC)loc_temp.csv", DataFrame)
    params.α_bar[:] .= saved_params.α_bar
    params.A_bar[:] .= saved_params.A_bar
    set_prices_and_params!(pd, params; q=saved_params.q, q_last=saved_params.q, ρ=saved_params.ρ, H_S_prior=0)
    return (;pd, params)
end

"Load spatially-calibrated parameters, but without aggregate parameters necessarily calibrated."
function load_spatially_calibrated_prices_params()
    df_calibrated = CSV.read(INTERMEDIATE_DATA_3_DIR*"$(N_LOC)loc.csv", DataFrame)
    df_params = CSV.read(SPATIAL_PARAMS_PATH, DataFrame)[1:N_LOC, :]
    df_moments = CSV.read(SPATIAL_MOMENTS_2020_PATH, DataFrame)[1:N_LOC, :]

    @assert df_params.PUMA2020 == df_moments.PUMA2020 == df_calibrated.PUMA2020

    # Prices #
    q = df_calibrated.q |> copy
    ρ = df_calibrated.ρ |> copy
    q_last = df_calibrated.q |> copy
    
    prices_vec = LocalPrices.(q, ρ, q_last) |> StructArray
    prices = pad_dims(prices_vec; left=3, ndims_new=4)

    # Params #
    spatial_vec = StructArray([Location(;
        GISMATCH = df_params.PUMA2020[i],
        # Set Externally
        A_g = df_params.A_g[i],
        α_g = df_params.α_g[i],
        δ_g = df_params.delta_g[i],
        elasticity = df_params.elasticity[i],
        δ_bar = df_moments.delta[i],
        # Calibrated
        H_S = df_calibrated.H_S[i],
        Π = df_calibrated.Π[i],
        A_bar = df_calibrated.A_bar[i],
        α_bar = df_calibrated.α_bar[i],
        # Derived
        A = df_calibrated.A_bar[i],
        α = df_calibrated.α_bar[i],
        δ = df_moments.delta[i],
    ) for i in 1:N_LOC])

    spatial = pad_dims(spatial_vec; left=3, ndims_new=4)
    params = Params(;spatial)
    
    return (;prices, params)
end


##################################
# Calibrate Aggregate Parameters #
##################################

# Housing Preference Weight γ #
#-----------------------------#
function get_rent_to_earnings_from_γ(pd, params, γ; kwargs...)
    params.γ = γ
    solve_steady_state_prices!(pd, params; kwargs...)
    return get_average_rent_to_earnings_workage(pd, params)
end

function calibrate_gamma(pd, params; rent_to_earnings_empirical=nothing, verbosity=0, rtol=1e-5, update_speed=0.1, γ_guess=params.γ, inner_rtol=2e-5, kwargs...)

    rent_to_earnings_empirical = something(rent_to_earnings_empirical, read_aggregate_moments("rent_to_earnings_workage"))

    err = Inf
    while rms(err) > rtol
        rent_to_earnings = get_rent_to_earnings_from_γ(pd, params, γ_guess; verbosity=verbosity-1, rtol=inner_rtol, kwargs...)
        err = (rent_to_earnings - rent_to_earnings_empirical)/rent_to_earnings_empirical
        γ_guess -= update_speed * err
        verbosity >= 1 && @show err, γ_guess
    end
    return (;pd, params, γ=γ_guess)
end

# Elasticity of Substitution between Housing and Goods σ #
#--------------------------------------------------------#
function get_eos_coef_empirical()
    df_moments = CSV.read(SPATIAL_MOMENTS_2020_PATH, DataFrame)
    df_params = CSV.read(SPATIAL_PARAMS_PATH, DataFrame)
    df = innerjoin(df_moments, df_params, on=:PUMA2020)
    
    # Convert units to 1000 USD/decade
    df.mean_rent = df.mean_rent / 1000 * 10 * 12
    df.mean_earn_workage = df.mean_labour_earnings_workingage / 1000 * 10
    
    # Compute rental housing and non-housing consumption price and quantity estimates 
    df.H_D_percap = df.H_S ./ df.population
    df.rent_level = df.mean_rent ./ df.H_D_percap
    df.income_after_rent = df.mean_earn_workage - df.mean_rent

    return eos_hat = get_eos_coef(df)
end

function get_eos_coef_from_σ(pd, params, σ; kwargs...)
    params.σ = σ
    solve_steady_state_prices!(pd, params; kwargs...)
    return get_eos_coef_model(pd, params)
end

function calibrate_σ(pd, params; eos_coef_empirical=nothing, verbosity=0, rtol=1e-5, update_speed=0.1, σ_guess=params.σ, inner_rtol=1e-4)

    eos_coef_empirical = something(eos_coef_empirical, get_eos_coef_empirical())

    err = Inf
    while rms(err) > rtol
        eos_coef = get_eos_coef_from_σ(pd, params, σ_guess; verbosity=verbosity-1, rtol=inner_rtol)
        err = (eos_coef - eos_coef_empirical)/eos_coef_empirical
        σ_guess -= update_speed * err
        verbosity >= 1 && @show err, σ_guess
    end
    return (;pd, params, σ=σ_guess)
end

# Fixed Utility Cost of Moving F_u_fixed #
#----------------------------------------#
function get_moving_share_from_F_u(pd, params, F_u; kwargs...)
    params.F_u_fixed = F_u
    solve_steady_state_prices!(pd, params; kwargs...)
    return get_share_moving(pd)
end

function calibrate_F_u_fixed(pd, params; moving_share_empirical=nothing, verbosity=0, rtol=1e-5, update_speed=1.0, F_u_guess=params.F_u_fixed, inner_rtol=2e-5, kwargs...)

    moving_share_empirical = something(moving_share_empirical, read_aggregate_moments("moving_share_decadal"))

    err = Inf
    while rms(err) > rtol
        moving_share = get_moving_share_from_F_u(pd, params, F_u_guess; verbosity=verbosity-1, rtol=inner_rtol)
        err = (moving_share - moving_share_empirical)/moving_share_empirical
        F_u_guess += update_speed * err
        verbosity >= 1 && @show err, F_u_guess
    end
    return (;pd, params, F_u_fixed=F_u_guess)
end

# Distance Utility Cost of Moving F_u_dist #
#------------------------------------------#
function get_avg_mig_distance_from_F_u_dist(pd, params, F_u_dist; kwargs...)
    params.F_u_dist = F_u_dist
    solve_steady_state_prices!(pd, params; kwargs...)
    return get_avg_mig_distance(pd)
end

function calibrate_F_u_dist(pd, params; avg_mig_distance_empirical=nothing, verbosity=0, rtol=1e-5, update_speed=1e-2, F_u_dist_guess=params.F_u_dist, inner_rtol=2e-5)

    avg_mig_distance_empirical = something(avg_mig_distance_empirical, read_aggregate_moments("avg_mig_distance"))

    err = Inf
    while rms(err) > rtol
        avg_mig_distance = get_avg_mig_distance_from_F_u_dist(pd, params, F_u_dist_guess; verbosity=verbosity-1, rtol=inner_rtol)
        err = (avg_mig_distance - avg_mig_distance_empirical)/avg_mig_distance_empirical
        F_u_dist_guess += update_speed * err
        verbosity >= 1 && @show err, F_u_dist_guess
    end
    return (;pd, params, F_u_dist=F_u_dist_guess)
end

# Maintenance Cost χ #
#--------------------#
function get_average_price_to_rent_from_χ(pd, params, χ; kwargs...)
    params.χ = χ
    solve_steady_state_prices!(pd, params; kwargs...)
    return get_average_price_to_rent(pd)
end

function calibrate_chi(pd, params; price_to_rent_empirical=nothing, verbosity=0, rtol=1e-5, update_speed=10.0, χ_guess=params.χ, inner_rtol=2e-5, kwargs...)

    price_to_rent_empirical = something(price_to_rent_empirical, read_aggregate_moments("price_to_rent_decadal"))

    err = Inf
    while rms(err) > rtol
        price_to_rent = get_average_price_to_rent_from_χ(pd, params, χ_guess; verbosity=verbosity-1, rtol=inner_rtol, kwargs...)
        err = (price_to_rent - price_to_rent_empirical)/price_to_rent_empirical
        χ_guess += update_speed * err
        verbosity >=1 && @show err, χ_guess
    end

    return (;pd, params, χ=χ_guess)
end

# Bequest Motive #
#----------------#
function get_old_relative_meanwealth_from_bequest_motive(pd, params, bequest_motive; kwargs...)
    params.bequest_motive = bequest_motive
    solve_steady_state_prices!(pd, params; kwargs...)
    
    return get_old_relative_meanwealth(pd)
end

function calibrate_bequest_motive(pd, params; old_relative_meanwealth_empirical=nothing, verbosity=0, rtol=1e-5, update_speed=1.0, bequest_motive_guess=params.bequest_motive, inner_rtol=2e-5, kwargs...)
    
    old_relative_meanwealth_empirical = something(old_relative_meanwealth_empirical, read_aggregate_moments("old_relative_meanwealth"))

    err = Inf
    while rms(err) > rtol
        old_relative_meanwealth = get_old_relative_meanwealth_from_bequest_motive(pd, params, bequest_motive_guess; verbosity=verbosity-1, rtol=inner_rtol, kwargs...)
        err = (old_relative_meanwealth - old_relative_meanwealth_empirical)/old_relative_meanwealth_empirical
        bequest_motive_guess -= update_speed * err
        verbosity >= 1 && @show err, bequest_motive_guess
    end

    return (;pd, params, bequest_motive=bequest_motive_guess)
end

# Location Preference Shock Variance ψ #
#--------------------------------------#
function get_rent_wage_coef_empirical()
    df_moments = CSV.read(SPATIAL_MOMENTS_2020_PATH, DataFrame)
    df_params = CSV.read(SPATIAL_PARAMS_PATH, DataFrame)
    df = innerjoin(df_moments, df_params, on=:PUMA2020)

    df.H_D_percap = df.H_S ./ df.population
    df.rent_level = df.mean_rent ./ df.H_D_percap
    
    return get_rent_wage_coef(df)
end

function get_rent_wage_coef_from_ψ(pd, params, ψ; kwargs...)
    params.ψ = ψ
    solve_steady_state_prices!(pd, params; kwargs...)
    return get_rent_wage_coef_model(pd, params)
end

function calibrate_ψ(pd, params; rent_wage_coef_empirical=nothing, verbosity=0, rtol=1e-5, update_speed=1.0, ψ_guess=params.ψ, inner_rtol=1e-4)

    rent_wage_coef_empirical = something(rent_wage_coef_empirical, get_rent_wage_coef_empirical())

    err = Inf
    while rms(err) > rtol
        rent_wage_coef = get_rent_wage_coef_from_ψ(pd, params, ψ_guess; verbosity=verbosity-1, rtol=inner_rtol)
        err = (rent_wage_coef - rent_wage_coef_empirical)/rent_wage_coef_empirical
        ψ_guess -= update_speed * err
        verbosity >= 1 && @show err, ψ_guess
    end
    return (;pd, params, ψ=ψ_guess)
end


################################
# Calibrate Everything Jointly #
################################

"""
Calibrate the model to the following targets, including spatial parameters in an inner loop.
    1. γ: rent_to_earnings
    2. σ: eos_coef
    3. F_u_fixed: moving_share_decadal
    4. F_u_dist: moving_distance_decadal
    5. χ: price_to_rent
    6. bequest_motive: old_relative_meanwealth
    7. ψ: rent_wage_coef
"""
function calibrate_model!(pd, params; verbosity=1, rtol=1e-2, update_speed=0.1, inner_rtol=1e-3, skip_params=[], save=false, save_intermediate=false)

    N_LOC < 100 && !(:F_u_dist in skip_params) && @warn "Cannot calibrate F_u_dist with less than 100 locations."

    update_γ = !(:γ in skip_params)
    update_σ = !(:σ in skip_params)
    update_F_u_fixed = !(:F_u_fixed in skip_params)
    update_F_u_dist = !(:F_u_dist in skip_params)
    update_χ = !(:χ in skip_params)
    update_bequest_motive = !(:bequest_motive in skip_params)
    update_ψ = !(:ψ in skip_params)

    rent_to_earnings_empirical = read_aggregate_moments("rent_to_earnings_workage")
    eos_coef_empirical = get_eos_coef_empirical()
    moving_share_empirical = read_aggregate_moments("moving_share_decadal")
    avg_mig_distance_empirical = N_LOC < 100 ? 112.3 : read_aggregate_moments("avg_mig_distance")
    price_to_rent_empirical = read_aggregate_moments("price_to_rent_decadal")
    old_relative_meanwealth_empirical = read_aggregate_moments("old_relative_meanwealth")
    rent_wage_coef_empirical = get_rent_wage_coef_empirical()

    calibration_err = Inf
    while calibration_err > rtol

        calibrate_spatial!(params, pd; verbosity=verbosity-1, rtol=inner_rtol, update_speed_base=0.5, save, save_intermediate)
        set_prices_and_params!(pd, params; H_S_prior=0)

        rent_to_earnings = get_average_rent_to_earnings_workage(pd, params)
        eos_coef = get_eos_coef_model(pd, params)
        moving_share = get_share_moving(pd)
        avg_mig_distance = get_avg_mig_distance(pd)
        price_to_rent = get_average_price_to_rent(pd)
        old_relative_meanwealth = get_old_relative_meanwealth(pd)
        rent_wage_coef = get_rent_wage_coef_model(pd, params)

        γ_err = (rent_to_earnings - rent_to_earnings_empirical) / rent_to_earnings_empirical
        σ_err = (eos_coef - eos_coef_empirical) / eos_coef_empirical
        F_u_fixed_err = (moving_share - moving_share_empirical) / moving_share_empirical
        F_u_dist_err = (avg_mig_distance - avg_mig_distance_empirical) / avg_mig_distance_empirical
        χ_err = (price_to_rent - price_to_rent_empirical) / price_to_rent_empirical
        bequest_motive_err = (old_relative_meanwealth - old_relative_meanwealth_empirical) / old_relative_meanwealth_empirical
        ψ_err = (rent_wage_coef - rent_wage_coef_empirical) / rent_wage_coef_empirical
        
        full_err_vec = []
        update_γ && push!(full_err_vec, γ_err^2)
        update_σ && push!(full_err_vec, σ_err^2)
        update_F_u_fixed && push!(full_err_vec, F_u_fixed_err^2)
        update_F_u_dist && push!(full_err_vec, F_u_dist_err^2)
        update_χ && push!(full_err_vec, χ_err^2)
        update_bequest_motive && push!(full_err_vec, bequest_motive_err^2)
        update_ψ && push!(full_err_vec, ψ_err^2)
        
        calibration_err = sqrt(mean(full_err_vec))
        verbosity >= 1 && @show calibration_err, full_err_vec

        calibration_err <= rtol && break

        update_γ && (params.γ -= update_speed * γ_err * 10)
        update_σ && (params.σ -= update_speed * σ_err / 10)
        update_F_u_fixed && (params.F_u_fixed += update_speed * F_u_fixed_err * 10)
        update_F_u_dist && (params.F_u_dist += update_speed * F_u_dist_err / 1)
        update_χ && (params.χ += update_speed * χ_err * 10)
        update_bequest_motive && (params.bequest_motive -= update_speed * bequest_motive_err * 100)
        update_ψ && (params.ψ -= update_speed * ψ_err / 10)

        if save_intermediate
            CSV.write(INTERMEDIATE_DATA_4_DIR*"$(N_LOC)loc_temp.csv", DataFrame(;
                γ = params.γ,
                σ = params.σ,
                F_u_fixed = params.F_u_fixed,
                F_u_dist = params.F_u_dist,
                χ = params.χ,
                bequest_motive = params.bequest_motive,
                ψ = params.ψ
            ))
        end
    end

    if save
        CSV.write(INTERMEDIATE_DATA_4_DIR*"$(N_LOC)loc.csv", DataFrame(;
            γ = params.γ,
            σ = params.σ,
            F_u_fixed = params.F_u_fixed,
            F_u_dist = params.F_u_dist,
            χ = params.χ,
            bequest_motive = params.bequest_motive,
            ψ = params.ψ
        ))
        CSV.write(INTERMEDIATE_DATA_4_DIR*"$(N_LOC)loc_untarged_moments.csv", DataFrame(;
            mean_wealth = get_mean_wealth(pd)
        ))
    end
    
    return (;pd, params)
end

function load_fully_calibrated_params(; temp=false)
    (;prices, params) = load_spatially_calibrated_prices_params()
    
    tempsuff = temp ? "_temp" : ""
    df_aggparams = CSV.read(INTERMEDIATE_DATA_4_DIR*"$(N_LOC)loc$tempsuff.csv", DataFrame)
    params.γ = df_aggparams.γ |> only
    params.σ = df_aggparams.σ |> only
    params.F_u_fixed = df_aggparams.F_u_fixed |> only
    params.F_u_dist = df_aggparams.F_u_dist |> only
    params.χ = df_aggparams.χ |> only
    params.bequest_motive = df_aggparams.bequest_motive |> only
    params.ψ = df_aggparams.ψ |> only
    
    pd = PeriodData(prices, params)
    return (;pd, params)
end
