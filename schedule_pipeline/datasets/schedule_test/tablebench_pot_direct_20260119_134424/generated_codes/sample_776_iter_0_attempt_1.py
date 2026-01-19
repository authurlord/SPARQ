import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'net assets at end of year' to numeric
df['net assets at end of year'] = pd.to_numeric(df['net assets at end of year'])

# Extract the last known net asset value and the increase trend
last_value = df['net assets at end of year'].iloc[-1]
increase_last_year = df['increase in net assets'].iloc[-1]

# Project the next year's net assets by assuming similar increase
projected_net_assets = last_value + increase_last_year
print(f"Final Answer: {projected_net_assets}")