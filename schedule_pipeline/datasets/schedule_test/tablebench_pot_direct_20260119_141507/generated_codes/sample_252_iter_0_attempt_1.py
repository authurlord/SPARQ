import pandas as pd

df = pd.read_csv('table.csv')
# Sort by points in descending order to get the top artists
df_sorted = df.sort_values(by='points', ascending=False)

# Highest points
max_points = df_sorted.iloc[0]['points']

# Top 3 artists' average points
top_3_avg = df_sorted.iloc[:3]['points'].mean()

# Difference
difference = max_points - top_3_avg
print(f"Final Answer: {difference}")