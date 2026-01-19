import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'passenger fleet' and 'current destinations' to numeric
df['passenger fleet'] = pd.to_numeric(df['passenger fleet'])
df['current destinations'] = pd.to_numeric(df['current destinations'])

# Calculate fleet utilization rate
df['fleet_utilization_rate'] = df['current destinations'] / df['passenger fleet']

# Find the airline with the highest fleet utilization rate
max_rate_airline = df.loc[df['fleet_utilization_rate'].idxmax(), 'airline / holding']

print(f"Final Answer: {max_rate_airline}")