import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'US Chart position' to numeric, coercing errors to NaN
df['US Chart position'] = pd.to_numeric(df['US Chart position'], errors='coerce')
# Drop rows where 'US Chart position' is NaN
df.dropna(subset=['US Chart position'], inplace=True)
# Calculate the average US chart position
average_chart_position = df['US Chart position'].mean()
print(f"Final Answer: {average_chart_position:.1f}")