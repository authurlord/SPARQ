import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'net assets at end of year' to numeric
df['net assets at end of year'] = pd.to_numeric(df['net assets at end of year'])

# Calculate the year-on-year increase in net assets
df['increase'] = df['net assets at end of year'].diff()

# Take the average increase over the available years
avg_increase = df['increase'].mean()

# Get the last known net asset value
last_value = df['net assets at end of year'].iloc[-1]

# Project the next year's value
projected_value = last_value + avg_increase

print(f"Final Answer: {projected_value:.0f}")