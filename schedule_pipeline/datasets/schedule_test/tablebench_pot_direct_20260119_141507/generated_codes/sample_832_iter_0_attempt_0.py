import pandas as pd

df = pd.read_csv('table.csv')
# Filter warships built before 1870
filtered_df = df[df['built year'] < 1870]
# Calculate standard deviation of horse-power
std_horse_power = filtered_df['horse - power'].std()
print(f"Final Answer: {std_horse_power:.1f}")