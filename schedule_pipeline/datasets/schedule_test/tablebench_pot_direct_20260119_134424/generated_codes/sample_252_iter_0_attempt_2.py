import pandas as pd

df = pd.read_csv('table.csv')
# Sort by points in descending order
df_sorted = df.sort_values(by='points', ascending=False)
# Get the top 3 artists' points
top_3_points = df_sorted['points'].head(3).astype(int)
# Calculate average of top 3
avg_top_3 = top_3_points.mean()
# Highest points
highest_points = df_sorted['points'].iloc[0]
# Difference
difference = highest_points - avg_top_3
print(f"Final Answer: {difference:.1f}")