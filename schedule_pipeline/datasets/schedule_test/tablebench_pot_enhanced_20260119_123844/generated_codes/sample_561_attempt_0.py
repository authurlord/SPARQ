import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 1942 to 1946
filtered_df = df[(df['Year'] >= '1942') & (df['Year'] <= '1946')]
# Convert 'US Chart position' to numeric, coercing errors to NaN
filtered_df['US Chart position'] = pd.to_numeric(filtered_df['US Chart position'], errors='coerce')
# Calculate the mean of valid positions
mean_chart_position = filtered_df['US Chart position'].mean()
print(f"Final Answer: {mean_chart_position:.1f}")