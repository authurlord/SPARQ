import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Year is 1944
df_1944 = df[df['Year'] == '1944']
# Extract US Chart position and clean non-numeric values
df_1944['US Chart position'] = pd.to_numeric(df_1944['US Chart position'], errors='coerce')
# Calculate mean, ignoring NaN values
avg_position = df_1944['US Chart position'].mean()
print(f"Final Answer: {avg_position:.1f}")