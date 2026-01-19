import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'points' column to numeric, coercing errors to NaN if any
df['points'] = pd.to_numeric(df['points'], errors='coerce')
# Calculate the average points, ignoring NaN values
average_points = df['points'].mean()
print(f"Final Answer: {average_points:.1f}")