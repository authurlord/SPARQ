import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'points' and 'laps' to numeric, handling invalid entries (like 'pe')
df['points'] = pd.to_numeric(df['points'], errors='coerce')
df['laps'] = pd.to_numeric(df['laps'], errors='coerce')

# Drop rows where either points or laps is NaN
df_clean = df.dropna(subset=['points', 'laps'])

# Calculate Points Per Lap
df_clean['points_per_lap'] = df_clean['points'] / df_clean['laps']

# Find the driver with the highest Points Per Lap
top_driver = df_clean.loc[df_clean['points_per_lap'].idxmax(), 'driver']
print(f"Final Answer: {top_driver}")