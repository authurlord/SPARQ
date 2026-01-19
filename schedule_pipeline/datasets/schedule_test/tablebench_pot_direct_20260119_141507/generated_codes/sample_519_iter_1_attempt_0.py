import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'passenger fleet' and 'current destinations' to numeric, replacing 'n / a' with NaN
df['passenger fleet'] = pd.to_numeric(df['passenger fleet'], errors='coerce')
df['current destinations'] = pd.to_numeric(df['current destinations'], errors='coerce')

# Remove rows where either value is NaN (invalid data)
df_clean = df.dropna(subset=['passenger fleet', 'current destinations'])

# Calculate fleet utilization rate
df_clean['fleet_utilization_rate'] = df_clean['current destinations'] / df_clean['passenger fleet']

# Find the airline with the highest utilization rate
max_utilization_airline = df_clean.loc[df_clean['fleet_utilization_rate'].idxmax(), 'airline / holding']

print(f"Final Answer: {max_utilization_airline}")