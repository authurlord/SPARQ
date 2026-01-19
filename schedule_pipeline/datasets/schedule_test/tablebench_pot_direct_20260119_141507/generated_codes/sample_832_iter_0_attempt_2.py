import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Filter warships built before 1870
filtered_df = df[df['built year'] < 1870]
# Extract horse-power values
horse_power = filtered_df['horse - power'].astype(float)
# Calculate standard deviation
std_horse_power = np.std(horse_power)
print(f"Final Answer: {std_horse_power:.1f}")