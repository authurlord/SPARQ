import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'points' column to integer
df['points'] = pd.to_numeric(df['points'])
# Find the highest points
highest_points = df['points'].max()
# Get top 3 artists by points
top_3_avg = df.nlargest(3, 'points')['points'].mean()
# Calculate the difference
difference = highest_points - top_3_avg
print(f"Final Answer: {difference:.1f}")