import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'points' column to numeric, coercing errors to NaN
df['points'] = pd.to_numeric(df['points'], errors='coerce')
# Calculate the mean of the 'points' column, ignoring NaN values
average_points = df['points'].mean()
print(f"Final Answer: {average_points:.1f}")