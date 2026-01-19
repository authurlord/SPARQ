import pandas as pd

df = pd.read_csv('table.csv')
# Filter warships built before 1870
filtered_df = df[df['built year'] < 1870]
# Extract horse-power values and calculate standard deviation
horse_power_std = filtered_df['horse - power'].std()
print(f"Final Answer: {horse_power_std}")