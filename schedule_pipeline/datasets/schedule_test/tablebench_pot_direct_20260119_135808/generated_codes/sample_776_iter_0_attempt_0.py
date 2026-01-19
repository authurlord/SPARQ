import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'net assets at end of year' to numeric
df['net assets at end of year'] = pd.to_numeric(df['net assets at end of year'])

# Calculate the year-over-year increase in net assets
df['increase'] = df['net assets at end of year'].diff()

# Compute the average annual increase
avg_increase = df['increase'].mean()

# Use the last known net asset value and add the average increase to project 2012/2013
last_value = df['net assets at end of year'].iloc[-1]
projected_value = last_value + avg_increase

print(f"Final Answer: {projected_value:.0f}")