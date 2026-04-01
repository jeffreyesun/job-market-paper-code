
"""
Compute average distance of migrations over 50 miles within the contiguous United States.
"""

using XLSX, DataFrames
using CSV
using Statistics

## Read in county-to-county migration flows
filename = "data/Census Bureau/county-to-county-2016-2020-ins-outs-nets-gross.xlsx"
xf = XLSX.readxlsx(filename)

dfs = Any[]

## Extract data from every sheet
for sheet_ind in 1:length(XLSX.sheetnames(xf))

    state = XLSX.sheetnames(xf)[sheet_ind]

    index_cols = xf[state*"!A4:D8000"]
    grossmig = xf[state*"!O4:O8000"]

    df = DataFrame(
        origin_state = index_cols[:, 1],
        origin_county = index_cols[:, 2],
        dest_state = index_cols[:, 3],
        dest_county = index_cols[:, 4],
        gross_migration = grossmig[:, 1],
    )
    push!(dfs, df)
end

df_all = vcat(dfs...)

## Clean missing and data types
for col in [:origin_state, :origin_county, :dest_state, :dest_county, :gross_migration]
    df_all = filter(row-> !ismissing(row[col]), df_all)
    df_all = filter(row-> (isa(row[col], Int64) || tryparse(Int64, row[col]) !== nothing), df_all)
    df_all[!, col] = Int64[isa(row[col], Int64) ? row[col] : parse(Int64, row[col]) for row in eachrow(df_all)]
end

## Drop Alaska (2), Hawaii (15), and Puerto Rico (72)
df_all = filter(row-> !(row.origin_state ∈ [2, 15, 72]) && !(row.dest_state ∈ [2, 15, 72]), df_all)

## Construct county IDs
df_all.county1 = df_all.origin_state .* 1000 .+ df_all.origin_county
df_all.county2 = df_all.dest_state .* 1000 .+ df_all.dest_county

## Merge in distance data
df_dist = CSV.read("data/Census Bureau/sf12010countydistancemiles.csv", DataFrame)

df_merged = innerjoin(df_all, df_dist, on=[:county1, :county2])
replace!(df_merged.gross_migration, missing=>0)

## Compute average move over 20 miles
mask = df_merged.mi_to_county .> 20
df_merged.gross_migration[mask]'df_merged.mi_to_county[mask] / sum(df_merged.gross_migration[mask])
# 400.8663117993612

## Compute continuous median of `mi_to_county` weighted by `gross_migration`
sorted_indices = sortperm(df_merged.mi_to_county[mask])
v_sorted = df_merged.mi_to_county[mask][sorted_indices]
w_sorted = df_merged.gross_migration[mask][sorted_indices]

median_plus_idx = findfirst(cumsum(w_sorted) .>= sum(w_sorted) / 2)
median_minus_idx = findlast(cumsum(w_sorted) .<= sum(w_sorted) / 2)
w_minus = w_sorted[median_minus_idx]
w_plus = w_sorted[median_plus_idx]

(v_sorted[median_minus_idx]*w_minus + v_sorted[median_plus_idx]*w_plus)/(w_minus + w_plus)
# 127.0020596299349

## Compute share of moves under 20 miles
(.!mask)'df_merged.gross_migration / sum(df_merged.gross_migration)
# 0.07729816021560111
