import pandas as pd

df = pd.read_csv('table.csv')
# Sort by points in descending order and take top 3
top_3 = df.sort_values(by='points', ascending=False).head(3)
# Calculate average points of top 3
avg_top_3_points = top_3['points'].mean()
# Highest points is the first row's points
highest_points = df['points'].max()
# Compute the difference
difference = highest_points - avg_top_3_points
print(f"Final Answer: {difference}")