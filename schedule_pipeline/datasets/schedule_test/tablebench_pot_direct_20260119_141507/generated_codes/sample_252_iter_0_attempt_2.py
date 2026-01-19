import pandas as pd

df = pd.read_csv('table.csv')
# Sort by points in descending order to get top performers
df_sorted = df.sort_values(by='points', ascending=False)
# Get the top 3 points
top_3_points = df_sorted['points'].head(3)
average_top_3 = top_3_points.mean()
max_points = df_sorted['points'].max()
difference = max_points - average_top_3
print(f"Final Answer: {difference:.0f}")