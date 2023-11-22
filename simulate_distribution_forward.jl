
##############
# Simulation #
##############

function apply_wealth_change_λ!(λ_post, λ_pre, wealth_post)
    wealth_post_tailsize = CartesianIndex(tail(size(wealth_post)))

    Threads.@threads for idx in CartesianIndices((N_Z, N_Hli, N_Hle, N_LOC))
        idx_k = min(idx, wealth_post_tailsize)
        @views convert_distribution!(
            λ_post[:,idx],
            λ_pre[:,idx],
            wealth_post[:,idx_k],
            WEALTH_GRID_FLAT
        )
    end
    return λ_post
end

function get_λ_postprice(λ_preprice, sim_prealloc)
    (;λ_postprice, wealth_postprice_k_preprice) = sim_prealloc
    return apply_wealth_change_λ!(λ_postprice, λ_preprice, wealth_postprice_k_preprice)
end

#get_λ_move(λ_presale, ad) = @. ad.λ_move = λ_presale * ad.P_move
#get_λ_nomove(λ_presale, λ_move, ad) = @. ad.λ_nomove = λ_presale .- λ_move

function get_λ_move_nomove(λ_presale, sim_prealloc)
    (;λ_move, λ_nomove, P_move) = sim_prealloc

    λ_move .= λ_presale .* P_move
    λ_nomove .= λ_presale .- λ_move

    return λ_move, λ_nomove
end

function get_λ_move_k_postmove(λ_move, sim_prealloc)
    (;wealth_postmove_k_presell, λ_move_k_postmove, kloc_scratch) = sim_prealloc

    λ_move_k_postmove .= 0

    Threads.@threads for loci=1:N_LOC
        vec = kloc_scratch[loci]
        for zi=1:N_Z
            for hi=CartesianIndices((N_Hli, N_Hle))
                @views convert_distribution!(
                    vec,
                    λ_move[:,zi,hi,loci],
                    wealth_postmove_k_presell[:,1,hi,loci],
                    WEALTH_GRID_FLAT # = wealth_presell_k_presell
                )
                @views λ_move_k_postmove[:,zi,1,1,loci] .+= vec
            end
        end
    end
    replace!(λ_move_k_postmove, NaN=>0)
    return λ_move_k_postmove
end

function get_λ_postmove(λ_move_k_postmove, sim_prealloc)
    (;eψV_move_k_postmove_tilde, eψFu_inv, eψV_postmove_tilde, λ_postmove, origin_weights) = sim_prealloc
    
    @. origin_weights = λ_move_k_postmove / eψV_move_k_postmove_tilde

    λ_postmove_mat = reshape(λ_postmove, N_K*N_Z, N_LOC)
    origin_mat = reshape(origin_weights, N_K*N_Z, N_LOC)
    mul!(λ_postmove_mat, origin_mat, eψFu_inv)

    λ_postmove .*= eψV_postmove_tilde
    N_LOC == 1 && replace!(λ_postmove, NaN=>0)
    return λ_postmove
end

function get_λ_premarket(λ_nomove, λ_postmove, sim_prealloc)
    (;λ_premarket) = sim_prealloc

    # Set the premarket population to the no-move population
    λ_premarket .= λ_nomove

    # Add the postmove population to the premarket population without housing
    @views λ_premarket[:,:,1:1,1:1,:] .+= λ_postmove

    return λ_premarket
end

function get_P_sell_buy_live!(P_sell, P_buy, V_sell, V_nosell)
    P_sell[:,:,1,:,:] .= 0
    #@threads
    for idx in CartesianIndices((N_K, N_Z, 2:N_Hli, N_Hle, N_LOC))
        eV_sell = exp(V_sell[idx])
        P_sell[idx] = eV_sell / (eV_sell + exp(V_nosell[idx]))
    end

    @threads for loci=1:N_LOC
        for ki=1:N_K, zi=1:N_Z, hlei=1:N_Hle
            @views P_buy[ki,zi,:,hlei,loci] .= exp.(V_nosell[ki,zi,:,hlei,loci])
            # Make it more invariant to the number of states by
            # treating "not buying" as a preference shock on part with all "buy" shocks
            P_buy[ki,zi,1,hlei,loci] *= N_Hli - 1
            @views P_buy[ki,zi,:,hlei,loci] ./= sum(P_buy[ki,zi,:,hlei,loci])
        end
    end

    return P_sell, P_buy
end

function get_P_sell_buy_let!(P_sell, P_buy, V_sell, V_nosell)
    PREF_POW = 5.0f0
    @assert !any(isnan, V_sell)
    @assert !any(isnan, V_nosell)

    P_sell[:,:,:,1,:] .= 0
    @threads for idx in CartesianIndices((N_K, N_Z, N_Hli, 2:N_Hle, N_LOC))
        eV_sell = exp(V_sell[idx])^PREF_POW
        P_sell[idx] = eV_sell / (eV_sell + exp(V_nosell[idx])^PREF_POW)
    end
    replace!(P_sell, NaN=>0)

    #@threads
    for loci=1:N_LOC
        for ki=1:N_K, zi=1:N_Z, hlii=1:N_Hli
            @views @. P_buy[ki,zi,hlii,:,loci] = V_nosell[ki,zi,hlii,:,loci]
            @views P_buy[ki,zi,hlii,:,loci] .-= mean(filter(isfinite, P_buy[ki,zi,hlii,:,loci]))
            @views @. P_buy[ki,zi,hlii,:,loci] = exp(P_buy[ki,zi,hlii,:,loci])^PREF_POW
            # Make it more invariant to the number of states by
            # treating "not buying" as a preference shock on part with all "buy" shocks
            P_buy[ki,zi,hlii,1,loci] *= N_Hle - 1
            @views P_buy[ki,zi,hlii,:,loci] ./= sum(P_buy[ki,zi,hlii,:,loci])
        end
    end
    replace!(P_buy, NaN=>0)

    return P_sell, P_buy
end

function get_λ_postbuy_live!(λ_postbuy, λ_prebuy, λ_presell, P_sell, P_buy)
    # presell -> prebuy
    # I'm being very careful here not to have to zero out λ_prebuy.
    # λ_prebuy .= 0
    # Start by adding all existing renters to the prebuy population
    @views λ_prebuy[:,:,1,:,:] .= λ_presell[:,:,1,:,:]
    @threads for loci=1:N_LOC
        for ki=1:N_K, zi=1:N_Z, hlii=2:N_Hli, hlei=1:N_Hle
            idx = CartesianIndex(ki,zi,hlii,hlei,loci)
            # Set the prebuy population for each state with housing to the portion of that state that doesn't sell
            λ_prebuy[idx] = λ_presell[idx] * (1 - P_sell[idx])
            # Add the portion of each state that sells to the prebuy population without housing
            λ_prebuy[ki,zi,1,hlei,loci] += λ_presell[idx] * P_sell[idx]
        end
    end

    # prebuy -> postbuy
    # Start by adding all existing homeowners to the postbuy population
    @views λ_postbuy[:,:,2:end,:,:] .= λ_prebuy[:,:,2:end,:,:]
    @views λ_postbuy[:,:,1,:,:] .= 0

    # For each state without housing, add the portion that buys to each the corresponding state
    # Note that choosing not to buy is encoded as buying hlii=1
    @threads for loci=1:N_LOC
        for ki=1:N_K, zi=1:N_Z, hlii=1:N_Hli, hlei=1:N_Hle
            idx = CartesianIndex(ki,zi,hlii,hlei,loci)
            λ_postbuy[idx] += λ_prebuy[ki,zi,1,hlei,loci] * P_buy[idx]
        end
    end

    return λ_postbuy
end

function get_λ_postbuy_let!(λ_postbuy, λ_prebuy, λ_presell, P_sell, P_buy)
    # presell -> prebuy
    # I'm being very careful here not to have to zero out λ_prebuy.
    # λ_prebuy .= 0
    # Start by adding all existing noninvestors to the prebuy population
    @views λ_prebuy[:,:,:,1,:] .= λ_presell[:,:,:,1,:]
    @threads for loci=1:N_LOC
        for ki=1:N_K, zi=1:N_Z, hlii=1:N_Hli, hlei=2:N_Hle
            idx = CartesianIndex(ki,zi,hlii,hlei,loci)
            # Set the prebuy population for each investor state to the portion of that state that doesn't sell
            λ_prebuy[idx] = λ_presell[idx] * (1 - P_sell[idx])
            # Add the portion of each state that sells to the prebuy population without investment
            λ_prebuy[ki,zi,hlii,1,loci] += λ_presell[idx] * P_sell[idx]
        end
    end

    # prebuy -> postbuy
    # Start by setting the postbuy population to the population of all carryover investors
    @views λ_postbuy[:,:,:,2:end,:] .= λ_prebuy[:,:,:,2:end,:]
    @views λ_postbuy[:,:,:,1,:] .= 0

    # For each noninvestor state, add the portion that buys to each the corresponding state
    # Note that choosing not to buy is encoded as buying hlei=1
    @threads for loci=1:N_LOC
        for ki=1:N_K, zi=1:N_Z, hlii=1:N_Hli, hlei=1:N_Hle
            idx = CartesianIndex(ki,zi,hlii,hlei,loci)
            λ_postbuy[idx] += λ_prebuy[ki,zi,hlii,1,loci] * P_buy[idx]
        end
    end

    return λ_postbuy
end

function get_λ_postbuy_live(λ_presell_live, sim_prealloc)
    (;P_sell_live, P_buy_live, V_sell_live, V_choosesell_let, λ_prebuy_live, λ_presell_let) = sim_prealloc
    λ_postbuy_live = λ_presell_let
    V_nosell_live = V_choosesell_let

    get_P_sell_buy_live!(P_sell_live, P_buy_live, V_sell_live, V_nosell_live)

    return get_λ_postbuy_live!(λ_postbuy_live, λ_prebuy_live, λ_presell_live, P_sell_live, P_buy_live)
end

function get_λ_postbuy_let(λ_presell_let, sim_prealloc)
    (;P_sell_let, P_buy_let, V_sell_let, V_income, λ_prebuy_let, λ_postmarket) = sim_prealloc
    λ_postbuy_let = λ_postmarket
    V_nosell_let = V_income

    get_P_sell_buy_let!(P_sell_let, P_buy_let, V_sell_let, V_nosell_let)

    return get_λ_postbuy_let!(λ_postbuy_let, λ_prebuy_let, λ_presell_let, P_sell_let, P_buy_let)
end

function get_λ_prec(λ_postmarket, params, sim_prealloc)
    (;wealth_postinc_k_preinc, λ_prec) = sim_prealloc

    Threads.@threads for loci=1:N_LOC
        # Convert λ_postmarket (over pre-income wealth)
        # to λ_prec (over post-income wealth)
        for idx=CartesianIndices((N_Z, N_Hli, N_Hle))
            @views convert_distribution!(
                λ_prec[:,idx,loci],
                λ_postmarket[:,idx,loci],
                wealth_postinc_k_preinc[:,idx,loci],
                WEALTH_GRID_FLAT)
        end
    end
    replace!(λ_prec, NaN=>0)

    return λ_prec
end

function get_λ_preshock(λ_prec, sim_prealloc)
    (;wealthi_postc_k_prec, λ_preshock) = sim_prealloc

    λ_preshock .= 0

    Threads.@threads for loci=1:N_LOC
        for idx=CartesianIndices((N_Z, N_Hli, N_Hle,loci:loci))
            for ki=1:N_K
                k1i = wealthi_postc_k_prec[ki,idx]
                λ_preshock[k1i,idx] += λ_prec[ki,idx]
            end
            # @views λ_preshock[[wealthi_postc_k_prec[:,idx],idx]] .+= λ_prec[:,idx]
        end
    end
    
    return λ_preshock
end

function get_λ_next(λ_postmarket, sim_prealloc)
    (;λ_next_mat, λ_next, λ_postmarket_perm) = sim_prealloc

    permutedims!(λ_postmarket_perm, λ_postmarket, (2, 1, 3, 4, 5))

    λ_postmarket_mat = reshape(λ_postmarket_perm, N_Z, N_K*N_Hli*N_Hle*N_LOC)
    mul!(λ_next_mat, Z_T_TRANSPOSE, λ_postmarket_mat)

    λ_next_perm = reshape(λ_next_mat, N_Z, N_K, N_Hli, N_Hle, N_LOC)
    permutedims!(λ_next, λ_next_perm, (2, 1, 3, 4, 5))

    return λ_next
end


################################
# Simulate Forward Full Period #
################################

# Evolution of State Distribution #
#---------------------------------#
function iterate_λ(λ_start, sim_prealloc, params)
    sim_prealloc.λ_start .= λ_start

    # Get population just after prices changes affect wealth
    λ_postprice = get_λ_postprice(λ_start, sim_prealloc)

    # Get population that moves
    ## by premove location and premove wealth
    λ_move, λ_nomove = get_λ_move_nomove(λ_postprice, sim_prealloc)
    ## by premove location and postmove wealth
    λ_move_k_postmove = get_λ_move_k_postmove(λ_move, sim_prealloc)
    ## by postmove location and postmove wealth
    λ_postmove = get_λ_postmove(λ_move_k_postmove, sim_prealloc)
    
    # Get population by post-move location and wealth
    λ_premarket = get_λ_premarket(λ_nomove, λ_postmove, sim_prealloc)

    # Get population by post-market, pre-income wealth
    λ_presell_let = get_λ_postbuy_live(λ_premarket, sim_prealloc)
    λ_postmarket = get_λ_postbuy_let(λ_presell_let, sim_prealloc)

    # Get population by post-income, pre-consumption wealth
    λ_prec = get_λ_prec(λ_postmarket, params, sim_prealloc)

    # Get population by post-consumption, pre-shock wealth
    λ_preshock = get_λ_preshock(λ_prec, sim_prealloc)

    # Get post-shock, end-of-period population
    λ_next = get_λ_next(λ_preshock, sim_prealloc)

    return λ_next
end

# Population Statistics #
#-----------------------#
"Sum over all households in a period. By default, don't sum over locations."
sum_perioddata(arrs::Vector{<:Array}, dims) = sum(a->sum(a; dims), arrs)
get_sum(pd::PeriodData, s, dims=(1,2,3,4)) = sum_perioddata(getproperty(pd, s), dims)
sumpop(period_data::PeriodData, dims=(1,2,3,4)) = get_sum(period_data, :λ_prec, dims)
function sumpop_agei(period_data::PeriodData, agei::Int; s=:λ_prec, dims=(2,3,4))
    return sum(getproperty(period_data, s)[agei]; dims)
end

#sum_perioddata([P_move.*λ_start for (P_move,λ_start)=zip(period_data.P_move, period_data.λ_start)], (1,2,3,4))

function get_weightedsum(value_vec::AbstractVector, weight_vec::AbstractVector, dims=(1,2,3,4))
    slices = [A.*B for (A,B)=zip(value_vec, weight_vec)]
    return sum_perioddata(slices, dims)
end

function get_weightedsum(value::AbstractArray{T,5}, weight_vec::AbstractVector, dims=(1,2,3,4)) where T
    slices = [value.*B for B=weight_vec]
    return sum_perioddata(slices, dims)
end

function get_weightedsum(ad::AgeData, value::Symbol, weight::Symbol, dims=(1,2,3,4))
    return sum(getproperty(ad, value) .* getproperty(ad, weight); dims=dims)
end

function get_weightedsum(pd::PeriodData, value, weight, dims=(1,2,3,4))
    return sum(agei->get_weightedsum(AgeData(pd, agei), value, weight, dims), 1:N_AGE)
end

function get_weightedmean(value_vec::Vector, weight_vec::Vector, dims=(1,2,3,4))
    return get_weightedsum(value_vec, weight_vec, dims) ./ sum_perioddata(weight_vec, dims)
end

function get_weightedmean(ad::AgeData, value::Symbol, weight::Symbol, dims=(1,2,3,4))
    return get_weightedsum(ad, value, weight, dims) ./ sum(getproperty(ad, weight); dims=dims)
end

function get_mean(period_data::PeriodData, value, weight, dims=(1,2,3,4))
    weightedsum = get_weightedsum(period_data, value, weight, dims)
    return weightedsum ./ get_sum(period_data, weight, dims)
end

# Interesting differences in mean wealth
function mean_wealth(period_data, loci)
    total_wealth = sumpop(period_data, (2,3,4))[:,1,1,1,loci]'WEALTH_GRID_FLAT
    return total_wealth / period_data.pop[loci]
end

"Compute demand for owner-occupied real estate."
function get_H_live(period_data::PeriodData)
    H_live_hlii = sumpop(period_data, (1,2,4))
    H_live_hlii_mat = reshape(H_live_hlii, N_Hli, N_LOC)
    H_live_flat = H_live_hlii_mat'H_LIVE_GRID_FLAT
    return reshape(H_live_flat, 1,1,1,1,N_LOC)
end

"Compute demand for rental real estate assets (*not* demand for rental housing)."
function get_H_let(period_data::PeriodData)
    H_let_hlei = sumpop(period_data, (1,2,3))
    H_let_hlei_mat = reshape(H_let_hlei, N_Hle, N_LOC)
    H_let_flat = H_let_hlei_mat'H_LET_GRID_FLAT
    return reshape(H_let_flat, 1,1,1,1,N_LOC)
end

"Fill in loc_grid with new equilibria."
function write_equilibrium_to_loc_grid!(period_data::PeriodData)
    (;loc_grid) = period_data
    loc_grid.H_D .= get_H_live(period_data) .+ get_H_let(period_data)
    loc_grid.pop .= sumpop(period_data)
    return period_data
end
function write_equilibrium_to_loc_grid!(path_data::PathData)
    for period_data in PeriodDataPath(path_data)
        write_equilibrium_to_loc_grid!(period_data)
    end
end

# All Together #
#--------------#
function simulate_H_forward!(period_data::PeriodData, params::Params)
    (;loc_grid) = period_data
    H = get_H.(loc_grid, loc_grid.q, params)
    loc_grid.H .= H
    return loc_grid
end

function simulate_H_forward_steady_state!(period_data::PeriodData, params::Params)
    (;loc_grid) = period_data
    H = get_H_construction.(loc_grid)
    loc_grid.H .= H
    return loc_grid
end

function simulate_λ_forward_steady_state!(period_data::PeriodData, params::Params)
    λ_start = get_λ_init_steady_state(period_data, params)

    # Simulate state distribution forward
    for agei in 1:N_AGE
        global agei_last = agei
        age_data = AgeData(period_data, agei)
        λ_start = iterate_λ(λ_start, age_data, params)
    end

    write_equilibrium_to_loc_grid!(period_data)
    return period_data
end

function simulate_forward_steady_state!(period_data::PeriodData, params::Params)
    simulate_H_forward_steady_state!(period_data, params)
    simulate_λ_forward_steady_state!(period_data, params)
    return period_data
end
