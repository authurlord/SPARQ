import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'horse - power' column to numeric
df['horse - power'] = pd.to_numeric(df['horse - power'], errors='coerce')
# Filter warships built before 1870
df_before_1870 = df[df['built year'] < 1870]
# Calculate standard deviation of horse-power for those warships
std_hp = df_before_1870['horse - power'].std()
print(f"Final Answer: {std_hp:.2f}")