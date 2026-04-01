
using DataFrames, CSV

df = CSV.read("data/IPUMS MOVEDIN/usa_00036.csv", DataFrame)
# Sample is retricted to ages 30-80

## Compute 10-year moving rate
df.over10y = [row.MOVEDIN in 5:7 for row in eachrow(df)]
df.under10y = [row.MOVEDIN in 1:4 for row in eachrow(df)]
df.movedin_na = [row.MOVEDIN == 0 for row in eachrow(df)]

share_over10y = sum(df.over10y.*df.PERWT)/sum(df.PERWT)
share_under10y = sum(df.under10y.*df.PERWT)/sum(df.PERWT)
share_na = sum(df.movedin_na.*df.PERWT)/sum(df.PERWT)

moving_rate = share_under10y./(share_under10y + share_over10y)
# 56.36%

## Compute rent-to-earnings ratio

df_rw = df[(df.AGE .< 60) .& (df.INCWAGE .> 0) .& (df.RENTGRS .> 0), :]

mean_rent = sum(df_rw.RENTGRS .* df_rw.PERWT) / sum(df_rw.PERWT)
mean_earnings = sum(df_rw.INCWAGE .* df_rw.PERWT) / sum(df_rw.PERWT)

mean_rent*12 / mean_earnings
# 0.339
