import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'points' column to numeric, coercing errors to NaN and then dropping invalid entries
df['points'] = pd.to_numeric(df['points'], errors='coerce')
# Calculate the mean of the points after conversion
average_points = df['points'].mean()
print(f"Final Answer: {average_points:.1f}")