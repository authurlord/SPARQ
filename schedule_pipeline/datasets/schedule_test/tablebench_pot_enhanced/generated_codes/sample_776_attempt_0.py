import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'net assets at end of year' to numeric
df['net assets at end of year'] = pd.to_numeric(df['net assets at end of year'])

# Calculate year-on-year increases
yearly_increase = df['net assets at end of year'].diff().dropna()

# Compute average annual increase
avg_increase = yearly_increase.mean()

# Get the last known net asset value
last_value = df['net assets at end of year'].iloc[-1]

# Project next year's value
projected_value = last_value + avg_increase

print(f"Final Answer: {projected_value:.0f}")