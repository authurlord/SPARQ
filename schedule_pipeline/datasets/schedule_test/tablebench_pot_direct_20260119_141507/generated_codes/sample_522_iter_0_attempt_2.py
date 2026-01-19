import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'points' and 'laps' to numeric, replacing invalid values (like 'pe') with NaN
df['points'] = pd.to_numeric(df['points'], errors='coerce')
df['laps'] = pd.to_numeric(df['laps'], errors='coerce')

# Remove rows with invalid points or laps
df_clean = df.dropna(subset=['points', 'laps'])

# Calculate Points Per Lap
df_clean['points_per_lap'] = df_clean['points'] / df_clean['laps']

# Find the driver with the highest Points Per Lap
max_driver = df_clean.loc[df_clean['points_per_lap'].idxmax(), 'driver']

print(f"Final Answer: {max_driver}")