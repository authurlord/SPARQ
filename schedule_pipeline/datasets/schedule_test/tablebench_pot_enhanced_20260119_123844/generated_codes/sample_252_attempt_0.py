import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'points' column to numeric
df['points'] = pd.to_numeric(df['points'])
# Find the highest points
max_points = df['points'].max()
# Get top 3 artists by points
top_3_avg = df.nlargest(3, 'points')['points'].mean()
# Calculate difference
difference = max_points - top_3_avg
print(f"Final Answer: {difference:.1f}")