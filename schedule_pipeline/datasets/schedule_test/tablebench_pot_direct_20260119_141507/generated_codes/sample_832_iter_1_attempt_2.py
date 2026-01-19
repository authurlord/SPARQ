import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert 'built year' to integer
df['built year'] = pd.to_numeric(df['built year'], errors='coerce')
# Filter warships built before 1870
filtered_df = df[df['built year'] < 1870]
# Extract horse-power values
horse_power_values = filtered_df['horse - power'].astype(float)
# Compute standard deviation
std_horse_power = np.std(horse_power_values)
print(f"Final Answer: {std_horse_power:.2f}")