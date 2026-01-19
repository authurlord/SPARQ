import pandas as pd

df = pd.read_csv('table.csv')
# Filter warships built before 1870
df_before_1870 = df[df['built year'] < 1870]
# Convert 'horse - power' to numeric
df_before_1870['horse - power'] = pd.to_numeric(df_before_1870['horse - power'])
# Calculate standard deviation
std_hp = df_before_1870['horse - power'].std()
print(f"Final Answer: {std_hp:.1f}")