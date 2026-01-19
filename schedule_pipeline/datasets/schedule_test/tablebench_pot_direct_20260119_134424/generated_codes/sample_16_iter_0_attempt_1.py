import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'US Chart position' to numeric, coercing errors to NaN
df['US Chart position'] = pd.to_numeric(df['US Chart position'], errors='coerce')
# Drop rows where the chart position is NaN
valid_positions = df['US Chart position'].dropna()
# Calculate the average
average_position = valid_positions.mean()
print(f"Final Answer: {average_position:.1f}")