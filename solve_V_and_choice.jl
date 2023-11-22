
#######################
# Construction Sector #
#######################

"Get housing prices, given by the inverse supply curve, conditional on positive construction."
get_q_construction(H, H_bar, Π, elasticity) = @. Π*(H/H_bar)^(1/elasticity)
"Get housing quantities, given by the supply curve, conditional on positive construction."
get_H_construction(q, H_bar, Π, elasticity) = @. H_bar*(q/Π)^elasticity

get_H_construction(loc::Location, q) = get_H_construction(q, loc.H_bar, loc.Π, loc.elasticity)
get_H_construction(loc::Location) = get_H_construction(loc, loc.q)

get_H(loc::Location, q, params::Params) = max(loc.H*(1-params.δ_dep), get_H_construction(loc, q))

###########################
# Household Forward Model #
###########################

# Utility #
#---------#
get_u(g, h, α, γ, σ, η) = (((1-γ)*(α*g/500)^(1-σ) + γ*(α*h)^(1-σ))^((1-η)/(1-σ)) - 1)/(1-η)

get_price_level(ρ, γ, σ) = (1-γ)^(1/σ) + γ^(1/σ) * ρ^((σ-1)/σ)
get_g_demand(c, ρ, γ, σ) = (1-γ)^(1/σ) * c / 500 / get_price_level(ρ/500, γ, σ)
get_h_demand(c, ρ, γ, σ) = (γ/(ρ/500))^(1/σ) * c / 500 / get_price_level(ρ/500, γ, σ)
get_indirect_u(c, ρ, α, γ, σ, η) = ( ((α*c/500) * get_price_level(ρ/500, γ, σ)^(σ/(1-σ)) )^(1-η) - 1)/(1-η)

function get_indirect_u(loc_grid, params)
    (;ρ, α) = loc_grid
    (;γ, σ, η) = params
    c_factor = @. (α*max(WEALTH_GRID - WEALTH_NEXT_GRID, 0)/500)^(1-η)
    price_level_exp = σ / (1-σ) * (1-η)
    price_level_factor = @. get_price_level(ρ/500, γ, σ)^price_level_exp
    return @. (c_factor * price_level_factor - 1) / (1-η)
end

function get_u_own(loc_grid, params)
    (;α) = loc_grid
    (;γ, σ, η) = params
    return @. get_u(max(WEALTH_GRID - WEALTH_NEXT_GRID, 0), H_LIVE_GRID, α, γ, σ, η)
end

# Income #
#--------#
"""
Within the model, it never makes sense for the household to have
positive bonds and positive mortgage, because the mortgage rate
is higher than the bond rate.
"""
function get_interest(wealth, h_live, h_let, loc, params)
    (;q) = loc
    (;r_m) = params

    net_bondholdings = wealth - q*(h_live + h_let)
    bond_interest = max(0, net_bondholdings)*r
    mortgage_interest = -min(0, net_bondholdings)*r_m

    return bond_interest - mortgage_interest
end

function get_noninterest_income(z, h_live, h_let, loc, params)
    (;A, δ, ρ) = loc
    (;χ_live, χ_let) = params
    
    income = A*z # earnings
    income += ρ*h_let # rental income
    expenses = h_live*(χ_live + δ) # Maintenance, owner-occupied
    expenses += h_let*(χ_let + δ) # Maintenance, rental
    
    return income - expenses
end

function get_income(wealth, z, h_live, h_let, loc, params)
    interest = get_interest(wealth, h_live, h_let, loc, params)
    noninterest_income = get_noninterest_income(z, h_live, h_let, loc, params)
    
    return interest + noninterest_income
end

# Wealth #
#--------#
"Get new wealth after the change in real estate price `q`."
function get_wealth_postprice(wealth, h_live, h_let, loc)
    (;q, q_last) = loc
    return wealth + (q - q_last)*(h_live + h_let)
end

function get_wealth_postinc(wealth, z, h_live, h_let, loc, params)
    wealth_postinc = wealth + get_income(wealth, z, h_live, h_let, loc, params)
    # I think wealth is supposed to be allowed to be negative coming out of here
    return wealth_postinc#max(wealth_postinc, 0)
end

function get_wealth_postsell(wealth_presell, h, loc, params)
    (;q) = loc
    (;ϕ) = params
    return wealth_presell - ϕ*q*h
end

function get_wealth_postmove(wealth_postsell, params)
    (;F_m) = params
    return wealth_postsell - F_m
end

function getκqh_own(h_live, h_let, loc, params)
    (;q) = loc
    (;κ) = params
    return κ*q*(h_live + h_let)
end


############################
# Household Value Function #
############################
"Reinterpolate a value function from after a wealth change to before."
function apply_wealth_change_V!(V_pre, V_post, wealth_post, left_extrap=Val(:linear))
    V_pre_tailsize = CartesianIndex(tail(size(V_pre)))
    V_post_tailsize = CartesianIndex(tail(size(V_post)))
    wealth_post_tailsize = CartesianIndex(tail(size(wealth_post)))

    Threads.@threads for idx=CartesianIndices((N_Z, N_Hli, N_Hle, N_LOC))
        @views reinterpolate!(
            V_pre[:,min(idx, V_pre_tailsize)],
            V_post[:,min(idx, V_post_tailsize)],
            WEALTH_GRID_FLAT,
            wealth_post[:,min(idx, wealth_post_tailsize)],
            left_extrap
        )
    end
    return V_pre
end

function get_V_bequest_i(wealth_next, h_live, h_let, loc, price_level_factor, params)
    (;α) = loc
    (;η, bequest_motive) = params

    wealth_postsell = get_wealth_postsell(wealth_next, h_live + h_let, loc, params)

    return wealth_postsell <= 0 ? -Inf : (
        α * bequest_motive * (wealth_postsell^(1-η) * price_level_factor - 1) / (1-η)
    )
end

function get_V_bequest!(prealloc, loc_grid, params)
    (;V_next) = prealloc
    (;ρ) = loc_grid
    (;γ, σ, η) = params
    price_level_exp = σ / (1-σ) * (1-η)
    price_level_factor = @. get_price_level(ρ, γ, σ)^price_level_exp
    return @. V_next = get_V_bequest_i(WEALTH_GRID, H_LIVE_GRID, H_LET_GRID, loc_grid, price_level_factor, params)
end

"Compute V_preshock. Get values in terms of z by multiplying along z' by z_T"
function get_V_preshock(prealloc, V_next)
    (;V_preshock, V_next_perm, V_preshock_perm) = prealloc

    # Put zi in front, so the entire thing is one big matrix multiplication
    permutedims!(V_next_perm, V_next, (2,1,3,4,5))

    V_next_mat = reshape(V_next_perm, N_Z, N_K*N_Hli*N_Hle*N_LOC)
    V_preshock_mat = reshape(V_preshock_perm, N_Z, N_K*N_Hli*N_Hle*N_LOC)

    mul!(V_preshock_mat, βZ_T, V_next_mat)

    permutedims!(V_preshock, V_preshock_perm, (2,1,3,4,5))
    return V_preshock
end


# V_consume
"""
    Convert any values below the borrowing constraint to -Inf.
    Must be applied whenever the household makes a decision that affects
    the borrowing constraint. I.e. saving or buying a house.
"""
function enforce_borrowing_constraint!(V, prealloc)
    V .+= prealloc.forbidden_states
    return V
end

"""
    Solve consumption problem.

Get pre-consumption value V_consume by maximizing
utility + continuation value
over all possible choices of continuation value.
Save the choices as wealthi_postc_k_prec.

V_consume(x) = max_{g,h} u(g,h,ℓ(x)) + V_postconsume(x'(x,g,h))
"""
function get_V_consume(V_postconsume, prealloc)
    (;V_consume, wealthi_postc_k_prec, u_indirect_rent, u_indirect_own) = prealloc

    Threads.@threads for loci=1:N_LOC
        # Compute optimal utility for renters
        u_rent = u_indirect_rent[loci]
        for idx in CartesianIndices((N_Z, 1, N_Hle))
            @views k1_argmax!(
                V_consume[:, idx, loci],
                wealthi_postc_k_prec[:, idx, loci],
                V_postconsume[:, idx, loci],
                u_rent
            )
        end

        # Compute optimal utility for homeowners
        for idx=CartesianIndices((N_Z, 2:N_Hli, N_Hle))
            hlii = idx[2]
            @views k1_argmax!(
                V_consume[:, idx, loci],
                wealthi_postc_k_prec[:, idx, loci],
                V_postconsume[:, idx, loci],
                u_indirect_own[hlii, loci]
            )
        end
    end

    return V_consume, wealthi_postc_k_prec
end


# V_income
"""
    Compute pre-income value, V_income.

Simply interpolate post-income value V_consume onto "post-income
wealth in terms of pre-income wealth," wealth_postinc_k_preinc.
"""
function get_V_income(V_consume, prealloc, loc_grid, params)
    (;agei, V_income, wealth_postinc_k_preinc) = prealloc
    X_agei = get_X_agei(loc_grid, agei)

    # Pre-sale, pre-move `wealth_presell`, in terms of post-sale, post-move `wealth_postmove`
    
    @. wealth_postinc_k_preinc = get_wealth_postinc(X_agei..., params)

    # We have V_consume in terms of wealth_postinc (= wealth_consume)
    # Put that in terms of wealth_preinc by resampling at wealth_postinc for each wealth_preinc
    Threads.@threads for idx=CartesianIndices((N_Z, N_Hli, N_Hle, N_LOC))
        @views reinterpolate!(
            V_income[:,idx], V_consume[:,idx], WEALTH_GRID_FLAT, wealth_postinc_k_preinc[:,idx],
            Val(-Inf)
        )
    end

    return V_income
end

# V_market
function get_V_choosesell_let(V_postbuy_let, prealloc)
    (;wealth_postsell_let_k_presell, V_choosebuy_let, V_sell_let, V_choosesell_let) = prealloc

    V_choosebuy_let .= maximum(V_postbuy_let; dims=H_LE_DIM)

    @views apply_wealth_change_V!(
        V_sell_let,
        V_choosebuy_let,
        wealth_postsell_let_k_presell,
        Val(:clip) # If households are bankrupt after selling, set their wealth to the minimum
    )

    V_nosell_let = V_postbuy_let
    @threads for idx=CartesianIndices(STATE_IDXs)
        V_choosesell_let[idx] = max(V_nosell_let[idx], V_sell_let[idx])
    end

    return V_choosesell_let
end

function get_V_choosesell_live(V_postbuy_live, prealloc)
    (;wealth_postsell_live_k_presell, V_choosebuy_live, V_sell_live, V_choosesell_live) = prealloc

    # Compute the value prior to buying new h_live, maximizing over choices
    V_choosebuy_live .= maximum(V_postbuy_live; dims=H_LI_DIM)

    # Compute the value of selling old h_live, using the value prior to buying new h_live
    # as the continuation value
    @views apply_wealth_change_V!(
        V_sell_live,
        V_choosebuy_live,
        wealth_postsell_live_k_presell,
        Val(:clip) # If households are bankrupt after selling, set their wealth to the minimum
    )

    # Compute the value prior to selling, maximizing over choices
    V_nosell_live = V_postbuy_live
    @threads for idx=CartesianIndices(STATE_IDXs)
        V_choosesell_live[idx] = max(V_nosell_live[idx], V_sell_live[idx])
    end

    validate_V_choosesell_live(V_choosesell_live)
    return V_choosesell_live
end

# V_move
"""
    Compute V_move in terms of k_postmove.

Because k_postmove is equal regardless of the destination, we can compute
V_move in terms of k_postmove by integrating over all possible destinations.
In fact, this can be a simple matrix multiplication. For each k_postmove, z,
we simply multiply the vector of exp(ψ * V_postmove(ℓ')) by the matrix of
exp(-ψ * F_u(ℓ, ℓ')).
"""
function get_V_move_k_postmove(V_postmove, prealloc, params)
    (; V_move_k_postmove, eψV_postmove_tilde, eψV_move_k_postmove_tilde, eψFu_inv, ψV_means) = prealloc
    (; ψ) = params

    ψV_means .= ψ.*mean(V_postmove; dims=(3,4,5))
    # Compute exp(ψ * V_postmove)
    @. eψV_postmove_tilde = fastexp(ψ*V_postmove - ψV_means)

    # Reshape everything into one big matrix
    eψV_postmove_mat = reshape(eψV_postmove_tilde, N_K*N_Z, N_LOC)
    eψV_move_k_postmove_mat = reshape(eψV_move_k_postmove_tilde, N_K*N_Z, N_LOC)
    
    mul!(eψV_move_k_postmove_mat, eψV_postmove_mat, eψFu_inv)

    # Recover V_move_k_postmove
    # This represents the location parameter of the Gumbel distribution, not the mean
    @. V_move_k_postmove = fastlog(eψV_move_k_postmove_tilde) + ψV_means# + EULER_GAMMA
    V_move_k_postmove ./= ψ

    return V_move_k_postmove
end

"Get post-move wealth in terms of pre-move wealth."
function get_wealth_postmove_from_presell(wealth_presell, h_live, h_let, loc, params)
    (;q) = loc
    (;ϕ, F_m) = params
    return wealth_presell - ϕ*q*(h_live + h_let) - F_m
end

"""
    Compute V_move (in terms of pre-sale wealth `k_presell`).
"""
function get_V_move(V_market, prealloc, params)
    (;wealth_postmove_k_presell, V_move) = prealloc

    # The value of moving, in terms of post-move location and wealth, is equal to the value of entering
    # the market with no owned real estate in the destination location.
    V_postmove = @view V_market[:,:,1:1,1:1,:]
    
    # Compute V_move by integrating over all possible destinations
    V_move_k_postmove = get_V_move_k_postmove(V_postmove, prealloc, params)

    # Resample V_move at wealth_presell for each wealth_postmove to get V_move in terms of wealth_presell
    apply_wealth_change_V!(V_move, V_move_k_postmove, wealth_postmove_k_presell, Val(-Inf))
    
    return V_move
end


# V_choosemove
"""
    Compute V_choosemove by comparing the value of moving with the value of not moving.

The probability of staying is computed
using the Gumbel CDF exp(-exp(-(x-μ)/β))
where
    x = V_nomove
    μ = log(Σᵢexp(ψV_moveᵢ))/ψ
      = V_move
    β = 1/ψ
so
    P(stay) = exp(-exp(-(V_nomove - V_move)*ψ))
            = exp(-eψV_move/eψV_nomove)
and the ex ante expected value is
      E[max(V_move, V_nomove)]
    = V_nomove + V_move + EULER_GAMMA/ψ - Ei(-exp(V_move*ψ))/ψ
    = V_nomove + V_move + EULER_GAMMA/ψ - Ei(-eψV_move)/ψ
"""
#=function get_V_choosemove_i(V_move, V_nomove, eψV_move_tilde, eψV_nomove_tilde, ψ)
    return V_move == -Inf ? V_nomove : (
    #V_move + EULER_GAMMA/ψ - fastexpinti(-eψV_move_tilde/eψV_nomove_tilde)/ψ
    # = V_move + EULER_GAMMA/ψ - expinti(fastlog(P_nomove))/ψ
    )
end=#
function get_V_choosemove_i(eψV_move_tilde, eψV_nomove_tilde, ψV_mean, ψ)
    # It should maybe technically be the above, but this is standard and a good approximation
    return (fastlog(eψV_move_tilde + eψV_nomove_tilde)+ψV_mean)/ψ
end

"Compute V_choosemove"
function get_V_choosemove(V_move, V_nomove, prealloc, params)
    (;eψV_move_tilde, eψV_nomove_tilde, ψV_means, P_move, V_choosemove) = prealloc
    (;ψ) = params

    Threads.@threads for idx in CartesianIndices(STATE_IDXs)
        kzi = CartesianIndex((idx[K_DIM], idx[Z_DIM], 1, 1, 1))

        eψV_move_tilde[idx] = fastexp(ψ*V_move[idx] - ψV_means[kzi])
        eψV_nomove_tilde[idx] = fastexp(ψ*V_nomove[idx] - ψV_means[kzi])

        P_move[idx] = eψV_move_tilde[idx] / (eψV_move_tilde[idx] + eψV_nomove_tilde[idx])
        #@.. except then you have to do the subtraction seprately!
        #@. P_move = 1 - fastexp(-eψV_move_tilde/eψV_nomove_tilde)
        V_choosemove[idx] = get_V_choosemove_i(eψV_move_tilde[idx], eψV_nomove_tilde[idx], ψV_means[kzi], ψ)
    end

    validate_V_choosemove(P_move, V_choosemove)
    return V_choosemove
end

# V_price
"""
Compute pre-price change value, V_price.

Simply interpolate post-price change value V_preprice onto "post-price
wealth in terms of pre-price wealth," wealth_postprice_k_preprice to
get "pre-price value in terms of pre-price wealth."
"""
function get_V_price(V_postprice, prealloc)
    (;V_price, wealth_postprice_k_preprice) = prealloc
    apply_wealth_change_V!(V_price, V_postprice, wealth_postprice_k_preprice)
    validate_V_price(V_price)
    return V_price
end


##################
# Full Algorithm #
##################


function solve_period!(prealloc, V_next, params)
    (;loc_grid) = prealloc
    prealloc.V_next .= V_next

    # V_shock
    V_preshock = get_V_preshock(prealloc, V_next)
    
    # V_consume and consumption/saving
    enforce_borrowing_constraint!(V_preshock, prealloc)
    V_consume, wealthi_postc_k_prec = get_V_consume(V_preshock, prealloc)
    
    # V_income (= V_postmarket)
    V_income = get_V_income(V_consume, prealloc, loc_grid, params)
    enforce_borrowing_constraint!(V_income, prealloc)
    
    # V_market and real estate asset choice
    V_choosesell_let = get_V_choosesell_let(V_income, prealloc)
    V_choosesell_live = get_V_choosesell_live(V_choosesell_let, prealloc)
        
    # V_move
    V_move = get_V_move(V_choosesell_live, prealloc, params)

    # V_choosemove
    V_nomove = V_choosesell_live
    V_choosemove = get_V_choosemove(V_move, V_nomove, prealloc, params)

    # V_price
    V_price = get_V_price(V_choosemove, prealloc)

    return prealloc
end


function solve_V_backward_steady_state!(period_data::PeriodData, params::Params)
    (;loc_grid) = period_data
    local V_next

    for agei in reverse(1:N_AGE)
        global agei_last = agei
        age_data = AgeData(period_data, agei)

        V_next = agei == N_AGE ? get_V_bequest!(age_data, loc_grid, params) : V_next

        age_data = solve_period!(age_data, V_next, params)

        V_next = age_data.V_price
    end

    return period_data
end
