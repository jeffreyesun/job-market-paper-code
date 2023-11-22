
function validate_V_choosemove(P_move, V_choosemove)
    @assert all(isfinite, V_choosemove)
    @assert all(>=(0), P_move)
    return nothing
end

function validate_V_price(V_price)
    @assert all(isfinite, V_price)
    return nothing
end

function validate_V_choosesell_live(V_choosesell_live)
    @assert all(isfinite, V_choosesell_live)
    return nothing
end

const LOC_INTERCEPT_FIELDS = [:Π, :H_bar, :A_bar, :α_bar, :δ_bar]
anyisnan(loc::Location) = any(s->isnan(getfield(loc, s)), fieldnames(Location))
function validate_loc(loc::Location)
    for pos_var in [:α, :ρ, :q, :q_last]
        @assert getfield(loc, pos_var) > 0
    end
    for s in filter(!in([LOC_INTERCEPT_FIELDS...,:H]), fieldnames(Location))
        @assert !isnan(getproperty(loc,s))
    end
    return nothing
end
validate_loc_grid(pd::PeriodData) = validate_loc.(pd.loc_grid)
validate_loc_grid_path(pd::PathData) = validate_loc.(pd.loc_grid_path)

# Interpolations #
#----------------#

validate_converted_distribution(y_new, y) = (@assert sum(y_new) ≈ sum(y); nothing)
