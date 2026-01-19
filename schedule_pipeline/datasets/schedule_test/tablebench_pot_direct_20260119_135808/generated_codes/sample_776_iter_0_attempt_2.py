import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'net assets at end of year' to numeric
df['net assets at end of year'] = pd.to_numeric(df['net assets at end of year'])

# Calculate year-over-year increase in net assets
df['increase'] = df['net assets at end of year'].diff()

# Compute average increase
avg_increase = df['increase'].mean()

# Get the last known net asset value
last_net_assets = df['net assets at end of year'].iloc[-1]

# Project next year's net assets
projected_net_assets = last_net_assets + avg_increase

print(f"Final Answer: {projected_net_assets:.0f}")