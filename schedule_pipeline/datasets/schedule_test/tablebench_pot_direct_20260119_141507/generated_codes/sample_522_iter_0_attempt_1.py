import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'points' and 'laps' to numeric, handling invalid entries like 'pe'
df['points'] = pd.to_numeric(df['points'], errors='coerce')
df['laps'] = pd.to_numeric(df['laps'], errors='coerce')

# Filter out rows where either points or laps are NaN
df_filtered = df.dropna(subset=['points', 'laps'])

# Calculate Points Per Lap
df_filtered['points_per_lap'] = df_filtered['points'] / df_filtered['laps']

# Find the driver with the highest Points Per Lap
top_driver = df_filtered.loc[df_filtered['points_per_lap'].idxmax(), 'driver']

print(f"Final Answer: {top_driver}")