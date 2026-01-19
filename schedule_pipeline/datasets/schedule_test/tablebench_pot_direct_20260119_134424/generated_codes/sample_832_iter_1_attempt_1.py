import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'built year' to integer for filtering
df['built year'] = pd.to_numeric(df['built year'], errors='coerce')
# Filter warships built before 1870
df_before_1870 = df[df['built year'] < 1870]
# Convert 'horse - power' to numeric, coercing errors to NaN
df_before_1870['horse - power'] = pd.to_numeric(df_before_1870['horse - power'], errors='coerce')
# Calculate standard deviation of horse-power
std_hp = df_before_1870['horse - power'].std()
print(f"Final Answer: {std_hp:.1f}")