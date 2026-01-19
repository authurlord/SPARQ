import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'net assets at end of year' to numeric
df['net assets at end of year'] = pd.to_numeric(df['net assets at end of year'])

# Calculate year-over-year growth rates
growth_rates = df['net assets at end of year'].pct_change().dropna()

# Calculate average growth rate
avg_growth_rate = growth_rates.mean()

# Get the last known net asset value
last_net_assets = df['net assets at end of year'].iloc[-1]

# Project next year's net assets
projected_net_assets = last_net_assets * (1 + avg_growth_rate)

print(f"Final Answer: {projected_net_assets:.0f}")