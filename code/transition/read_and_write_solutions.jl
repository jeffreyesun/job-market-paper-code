
function write_transition_moments(td::Vector{<:PeriodData}, pp::Vector{<:Params}; save=true, suffix="")
    df_slices = get_dataframe_slices.(td, pp)
    df_panel = reduce(vcat, slice.df_panelslice for slice=df_slices)
    df_series = reduce(vcat, slice.df_seriesslice for slice=df_slices)

    if save
        CSV.write(TRANSITION_MOMENTS_DIRPATH*"transition_panel_$(N_LOC)loc$suffix.csv", df_panel)
        CSV.write(TRANSITION_MOMENTS_DIRPATH*"transition_series_$(N_LOC)loc$suffix.csv", df_series)
    end
    return (;df_panel, df_series)
end

function read_transition_prices(pp; suffix="", T=length(pp))
    df = CSV.read(TRANSITION_MOMENTS_DIRPATH*"transition_panel_$(N_LOC)loc$suffix.csv", DataFrame)
    q = eachcol(reshape(df.q, (N_LOC, T)))
    q_last = eachcol(reshape(df.q_last, (N_LOC, T)))
    ρ = eachcol(reshape(df.rent, (N_LOC, T)))
    return (;q, q_last, ρ)
end
