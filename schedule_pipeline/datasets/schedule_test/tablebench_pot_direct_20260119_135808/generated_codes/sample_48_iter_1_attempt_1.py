import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'points' column to numeric, handling any potential non-numeric entries
df['points'] = pd.to_numeric(df['points'], errors='coerce')
# Calculate the average points, ignoring any NaN values
average_points = df['points'].mean()
print(f"Final Answer: {average_points:.1f}")