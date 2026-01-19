import pandas as pd

df = pd.read_csv('table.csv')
# Sort by points in descending order and get the top two values
points_sorted = df['points'].sort_values(ascending=False)
highest_points = points_sorted.iloc[0]
second_highest_points = points_sorted.iloc[1]
difference = highest_points - second_highest_points
print(f"Final Answer: {difference}")