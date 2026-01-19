import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 1942 to 1946
df_filtered = df[(df['Year'] >= '1942') & (df['Year'] <= '1946')]
# Convert 'US Chart position' to numeric, coercing errors to NaN
df_filtered['US Chart position'] = pd.to_numeric(df_filtered['US Chart position'], errors='coerce')
# Calculate average US Chart position
average_position = df_filtered['US Chart position'].mean()
print(f"Final Answer: {average_position:.1f}")