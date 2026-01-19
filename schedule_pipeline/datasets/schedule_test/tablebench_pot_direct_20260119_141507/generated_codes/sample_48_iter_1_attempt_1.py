import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'points' column to numeric, coercing errors to NaN
df['points'] = pd.to_numeric(df['points'], errors='coerce')
# Calculate the mean of points, ignoring any NaN values
mean_points = df['points'].mean()
print(f"Final Answer: {mean_points:.1f}")