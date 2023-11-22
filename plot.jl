
using GLMakie
using Colors
using GeometryBasics

##########
# Helper #
##########

newdisplay(fig) = display(GLMakie.Screen(), fig)

##########
# Plot V #
##########

zi_low, zi_med, zi_high = 1,2,3
hlii_low, hlii_med, hlii_high = 2,3,4
hlei_low, hlei_med, hlei_high = 2,3,4

const PLOT_LOG_OFFSET = 1e-1
log1pp(x) = fastlog(x + PLOT_LOG_OFFSET)
Makie.inverse_transform(::typeof(log1pp)) = x -> fastexp(x) - PLOT_LOG_OFFSET
Makie.defaultlimits(::typeof(log1pp)) = (0, 1e3)
Makie.defined_interval(::typeof(log1pp)) = Makie.OpenInterval(-PLOT_LOG_OFFSET, Inf)

ax_log1pp(fig) = Axis(fig[1,1]; xscale=log1pp, xticks=[0, 10.0.^(-1:4)...])

function plotband!(ax, V, hlii=1, hlei=1, loci=1; label="", kwargs...)
    v_lowz = V[:,zi_low,hlii,hlei,loci]
    v_highz = V[:,zi_high,hlii,hlei,loci]
    
    i0_lz = findfirst(>(-Inf), v_lowz)
    i0_hz = findfirst(>(-Inf), v_highz)
    isnothing(i0_lz) && isnothing(i0_hz) && return nothing

    if i0_lz == i0_hz
        i0 = i0_lz
        b = band!(ax, WEALTH_GRID_FLAT[i0:end], v_lowz[i0:end], v_highz[i0:end]; label, kwargs...)
        # Set the transparency of the band to 0.5
        #b.transparency = 0.5
    else
        i0 = min(i0_lz, i0_hz)
        b = band!(ax, WEALTH_GRID_FLAT[i0:end], v_lowz[i0:end], v_highz[i0:end]; label, kwargs...)
        color = b.color
        lines!(ax, WEALTH_GRID_FLAT[i0_lz:i0], v_lowz[i0_lz:i0]; kwargs...)
        lines!(ax, WEALTH_GRID_FLAT[i0_hz:i0], v_highz[i0_hz:i0]; kwargs...)
    end
    return b
end

function plotline!(ax, V, hlii=1, hlei=1; loci=1, zi=zi_med, kwargs...)
    idx = CartesianIndices(to_indices(V, Tuple(i === Colon() ? i : i:i for i=(Colon(),zi,hlii,hlei,loci))))
    v = vec(sum(V[idx]; dims=(2,3,4,5)))
    global V_save = V
    global idx_save = idx
    global v_save = v
    return lines!(ax, WEALTH_GRID_FLAT, v; kwargs...)
end

function plot_V(_V;
        hlii=nothing, hlei=nothing, sum_margins=false, loci=1, hlii_comp=hlii_med, hlei_comp=hlei_med,
        fig=Figure(), ax=ax_log1pp(fig), legend=true, band=true, field=nothing, kwargs...
    )
    @assert ndims(_V) in (5,6)
    V = _V[:,:,:,:,:,1]

    hlii = sum_margins ? Colon() : something(hlii, 1)
    hlei = sum_margins ? Colon() : something(hlei, 1)
    
    size(V, 2) > 1 && band && plotband!(ax, V; label="renter", color=(:blue,0.2), kwargs...)
    zi = min(zi_med, size(V, 2))

    if size(V, H_LI_DIM) > 1
        plotline!(ax, V, hlii_comp, hlei; zi, loci, label="owner", color=:orange, kwargs...)
    end
    if size(V, H_LE_DIM) > 1
        plotline!(ax, V, hlii, hlei_comp; zi, loci, label="investor", color=:green, kwargs...)
    end

    plotline!(ax, V, hlii, hlei; zi, loci, label="renter", color=:blue, kwargs...)

    legend && (fig[1,2] = Legend(fig, ax, ""; framevisible = false, merge=true))

    return fig
end

function line(v; print=true)
    print && println(round.(v; digits=6))
    fig = Figure()
    ax = ax_log1pp(fig)
    lines!(ax, WEALTH_GRID_FLAT, v)
    xlims!(ax, 0, maximum(WEALTH_GRID_FLAT))
    ylims!(ax, 0, maximum(v))
    return fig
end

function explore(data, default_field=:V_preshock; sum_margins=false)
    fig = Figure()
    ax = ax_log1pp(fig)
        
    plot_params = Ref((;
        field=default_field, loci=1, sum_margins=sum_margins,
        hlii=1, hlei=1, hlii_comp=hlii_med, hlei_comp=hlii_med,
    ))
    
    plot_V(getproperty(data, plot_params[].field); fig, ax)

    function plot_field(new_params)
        # Empty the figure
        empty!(ax)
        for leg in fig.content
            leg isa Legend && delete!(leg) 
        end

        # Plot the previous field in dotted lines
        (; field) = plot_params[]
        plot_V(getproperty(data, field); plot_params[]..., fig, ax, legend=false, band=false, linestyle=:dot)

        # Plot the current field in solid lines
        plot_params[] = new_params
        V = getproperty(data, new_params.field)
        plot_V(V; new_params..., fig, ax)

        # Set axis limits
        xlims!(ax, 0, maximum(WEALTH_GRID_FLAT))
        ylims!(ax, 0, maximum(V))
    end

    function register_callback(menu, param_name)
        on(menu.selection) do param
            #println("Setting $param_name to $param")
            new_params = NamedTuple(Dict(param_name => param))
            plot_field(merge(plot_params[], new_params))
        end
        return menu
    end

    fields = subfieldnames(data)
    field_options = zip(string.(fields), fields)
    menu_field = register_callback(Menu(fig; options=field_options, default=string(default_field), width=200), :field)
    menu_hlii = register_callback(Menu(fig, options=1:N_Hli, default=1), :hlii)
    menu_hlei = register_callback(Menu(fig, options=1:N_Hle, default=1), :hlei)
    menu_sum = register_callback(Menu(fig; options=[("no",false), ("yes",true)], default="no"), :sum_margins)
    menu_loci = register_callback(Menu(fig, options=1:20, default=1), :loci)
    menu_hlii_comp = register_callback(Menu(fig, options=1:N_Hli, default=hlii_med), :hlii_comp)
    menu_hlei_comp = register_callback(Menu(fig, options=1:N_Hle, default=hlei_med), :hlei_comp)

    fig[2, 1] = hgrid!(
        Label(fig, "Field", height=nothing),       menu_field,
        Label(fig, "h_lii", height=nothing),       menu_hlii,
        Label(fig, "h_lei", height=nothing),       menu_hlei,
        Label(fig, "Sum Margins", height=nothing), menu_sum,
        ; tellwidth = false, height = 100
    )
    fig[3, 1] = hgrid!(
        Label(fig, "loci", height = nothing),    menu_loci,
        Label(fig, "hlii_comp", height=nothing), menu_hlii_comp,
        Label(fig, "hlei_comp", height=nothing), menu_hlei_comp,
        ; tellwidth = false, height = 100
    )

    return fig
end
explore(pd::PeriodData, agei=1) = explore(AgeData(pd, agei))
explore_sum(args...) = explore(args...; sum_margins=true)

####################
# Plot Equilibrium #
####################

function plot_rental_market(
        period_data::PeriodData, params::Params;
        ρ_min=nothing, ρ_max=nothing, ρ_n=100, loci=1
    )
    (;λ_prec, loc_grid) = period_data
    (;ρ) = loc_grid
    ρ_min = something(ρ_min, ρ[loci]-10)
    ρ_max = something(ρ_max, ρ[loci]+10)
    ρ_init = ρ[loci]

    ρ_loc_flat = loc_grid.ρ
    ρ_samples = range(ρ_min, ρ_max; length=ρ_n)
    H_rent_vec = similar(ρ_samples)
    H_let_vec = similar(ρ_samples)
    pop_vec = similar(ρ_samples)

    for (i, ρ) in enumerate(ρ_samples)
        @show i
        ρ_loc_flat[loci] = ρ
        loc_grid.ρ .= ρ_loc_flat
        period_data = precompute!(period_data, params)

        period_data = solve_household_problem_steady_state!(period_data, params)
        H_let, H_rent = get_rental_market_clearing(period_data, params)
        H_rent_vec[i] = H_rent[loci]
        H_let_vec[i] = H_let[loci]
        pop_vec[i] = sum(sum(pop[:,:,:,:,loci]) for pop in λ_prec)
    end
    ρ[loci] = ρ_init
    solve_household_problem_steady_state!(period_data, params; ρ)

    fig = Figure()
    ax = Axis(fig[1,1])
    lines!(ax, ρ_samples, H_rent_vec, label="H_rent")
    lines!(ax, ρ_samples, H_let_vec, label="H_let")
    lines!(ax, ρ_samples, pop_vec, label="pop")
    axislegend(ax)
    return fig
end

function get_real_estate_market_lines(
        period_data::PeriodData, params::Params;
        q_min=nothing, q_max=nothing, q_n=100, loci=1
    )
    (;loc_grid) = period_data
    (;q) = loc_grid
    q_min = something(q_min, loc_grid.q[loci]-5)
    q_max = something(q_max, loc_grid.q[loci]+5)
    
    q_init = q[loci]
    q_samples = range(q_min, q_max; length=q_n)
    H_D_vec = similar(q_samples)
    H_S_vec = similar(q_samples)
    
    for (i, q_i) in enumerate(q_samples)
        @show i
        q[loci] = q_i

        solve_household_problem_steady_state!(period_data, params; q)

        H_D, H_S = get_real_estate_market_clearing(period_data, params)
        H_D_vec[i] = H_D[loci]
        H_S_vec[i] = H_S[loci]
    end
    q[loci] = q_init
    solve_household_problem_steady_state!(period_data, params; q) #if reset

    return q_samples, H_D_vec, H_S_vec
end

function plot_real_estate_market(
        period_data::PeriodData, params::Params;
        kwargs...
    )
    q_samples, H_D_vec, H_S_vec = get_real_estate_market_lines(period_data, params; kwargs...)

    fig = Figure()
    ax = Axis(fig[1,1]; xlabel="Housing price q", ylabel="Housing Quantity H")
    lines!(ax, q_samples, H_D_vec, label="H_D")
    lines!(ax, q_samples, H_S_vec, label="H_S")
    axislegend(ax)
    return fig
end

function plot_α_sensitivity(period_data, params; α_min=nothing, α_max=nothing, α_n=100, loci=1)
    (;loc_grid) = period_data
    α_min = something(α_min, loc_grid.α[loci]-0.1)
    α_max = something(α_max, loc_grid.α[loci]+0.1)

    α_loc_flat = loc_grid.α
    α_samples = range(α_min, α_max; length=α_n)
    pop_sim_vec = similar(α_samples)

    for (i, α) in enumerate(α_samples)
        @show i
        α_loc_flat[loci] = α
        loc_grid.α .= α_loc_flat
        period_data = precompute!(period_data, params)

        period_data = solve_household_problem_steady_state!(period_data, params)
        pop_sim_vec[i] = sumpop(period_data)[loci]
    end

    fig = Figure()
    ax = Axis(fig[1,1]; xlabel="Local Amenities α", ylabel="Location i Population")
    lines!(ax, α_samples, pop_sim_vec, label="pop_sim")
    lines!(ax, α_samples, fill(1.4, α_n), label="pop_data")
    axislegend(ax)
    return fig
end


##################
# Plot Decisions #
##################

"Plot V_postmarket, conditional on each housing purchase decision"
function plot_V_hli(age_data; zi=zi_med, hlei=1, loci=1)
    V = age_data.V_income[:,zi,2:end,hlei,loci]
    fig = Figure()
    ax = ax_log1pp(fig)
    colors = range(colorant"red", stop=colorant"blue", length=N_Hli-1)
    for hlii=1:N_Hli-1
        lines!(ax, WEALTH_GRID_FLAT, V[:,hlii], label="hlii=$hlii", color=colors[hlii])
    end
    fig[1,2] = Legend(fig, ax)
    xlims!(ax, findfirst(>(-Inf), maximum(V; dims=2))[1], WEALTH_GRID_FLAT[end])
    ylims!(ax, minimum(filter(>(-Inf), V)), maximum(V))
    return fig
end

"Plot V_postmarket, conditional on each rental real estate purchase decision"
function plot_V_hle(age_data; zi=zi_med, hlii=1, loci=1)
    V = age_data.V_income[:,zi,hlii,2:end,loci]
    fig = Figure()
    ax = ax_log1pp(fig)
    colors = range(colorant"red", stop=colorant"blue", length=N_Hle-1)
    for hlei=1:N_Hle-1
        lines!(ax, WEALTH_GRID_FLAT, V[:,hlei], label="hlei=$hlei", color=colors[hlei])
    end
    fig[1,2] = Legend(fig, ax)
    xlims!(ax, findfirst(>(-Inf), maximum(V; dims=2))[1], WEALTH_GRID_FLAT[end])
    ylims!(ax, minimum(filter(>(-Inf), V)), maximum(V))
    return fig
end

############################
# Plot Location Conditions #
############################

function explore(loc_grid::LocGrid, n_loc=length(loc_grid))
    f = Figure()
    ax = Axis(f[1, 1])

    scatter!(ax, 1:n_loc, vec(loc_grid.α), label="α", marker=:circle)
    scatter!(ax, 1:n_loc, vec(loc_grid.A), label="A", marker=:rect)
    scatter!(ax, 1:n_loc, vec(loc_grid.ρ)./100, label="ρ", marker=:diamond)
    scatter!(ax, 1:n_loc, vec(loc_grid.q)./100, label="q", marker=:cross)
    scatter!(ax, 1:n_loc, vec(loc_grid.q_last)./100, label="q_last", marker=:dagger)
    scatter!(ax, 1:n_loc, vec(loc_grid.pop), label="pop", marker='p')
    scatter!(ax, 1:n_loc, vec(loc_grid.H), label="H", marker='H')
    scatter!(ax, 1:n_loc, vec(loc_grid.H_D), label="H_D", marker='D')
    axislegend(ax)
    return f
end

################################
# Plot Population Distribution #
################################

POPULATION_STAGES = [:λ_start, :λ_postprice, :λ_move, :λ_nomove, :λ_premarket, :λ_postmarket, :λ_prec, :λ_preshock, :λ_next]
function plot_household_transition(ad::AgeData, loci::Int; zi=zi_med)
    h_live_hm = [squeeze(sum(getproperty(ad, stage)[:,zi:zi,:,:,loci]; dims=(4,))) for stage=POPULATION_STAGES]
    h_let_hm = [squeeze(sum(getproperty(ad, stage)[:,zi:zi,:,:,loci]; dims=(3,))) for stage=POPULATION_STAGES]
    fig = Figure()
    for (i, stage) in enumerate(POPULATION_STAGES)
        ax_live = Axis(fig[1,i]; ylabel=(i==1 ? "h_live" : ""))
        ax_let = Axis(fig[2,i]; xlabel=string(stage), ylabel=(i==1 ? "h_let" : ""))
        heatmap!(ax_live, h_live_hm[i])
        heatmap!(ax_let, h_let_hm[i])
    end
    return fig
end

function plot_household_transition(pd::PeriodData, agei::Int, loci::Int, params::Params; zi=zi_med)
    ad = get_age_data_solved!(pd, agei, params)
    return plot_household_transition(ad, loci; zi)
end

#####################
# Manhattan Diagram #
#####################

function manhattan_diagram()
    f = Figure()
    ax = Axis3(f[1, 1], aspect=(0.5,0.5,1), perspectiveness=0.75)
    rects = GeometryBasics.mesh.(Rect3.(0,0,0,2,1,vec(H_live_mean)))
    mesh!(ax, rects[51])

    x = zeros(N_K)
    y = 1:N_K
    z = zeros(N_K)
    # x = age

    fig = Figure(resolution=(1200, 800), fontsize=26)
    ax = Axis3(fig[1, 1]; aspect=(1,1,1), elevation=π/6, perspectiveness=0.5)
    rectMesh = Rect3f(Vec3f(-0.5, -0.5, 0), Vec3f(1, 1, 1))
    meshscatter!(
        ax, x, y, 0 * z; marker=rectMesh, color=range(0,0.1,length=N_K),
        markersize=Vec3f.(1, 1, vec(H_live_mean)), shading=true
    )
    GLMakie.zlims!(ax, 0, 13)
    fig

    mesh!.(Ref(ax), points, rects)

    H_live_tot = sum(λ_prec_loci .* H_LIVE_GRID; dims=(2,3,4))
    pop_tot = sum(λ_prec_loci; dims=(2,3,4))
    H_live_tot
end
