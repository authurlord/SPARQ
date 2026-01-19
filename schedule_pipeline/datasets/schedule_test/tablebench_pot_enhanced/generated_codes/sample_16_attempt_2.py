import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'US Chart position' to numeric, coercing errors to NaN
df['US Chart position'] = pd.to_numeric(df['US Chart position'], errors='coerce')
# Calculate the mean, ignoring NaN values
mean_chart_position = df['US Chart position'].mean()
print(f"Final Answer: {mean_chart_position:.1f}")