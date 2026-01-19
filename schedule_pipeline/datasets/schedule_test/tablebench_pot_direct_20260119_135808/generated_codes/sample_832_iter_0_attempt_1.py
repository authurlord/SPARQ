import pandas as pd

df = pd.read_csv('table.csv')
# Filter warships built before 1870
df_before_1870 = df[df['built year'] < 1870]
# Convert 'horse-power' to numeric and calculate standard deviation
std_hp = df_before_1870['horse - power'].astype(float).std()
print(f"Final Answer: {std_hp:.2f}")