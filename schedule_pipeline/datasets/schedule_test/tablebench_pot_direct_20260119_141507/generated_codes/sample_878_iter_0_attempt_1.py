import pandas as pd

df = pd.read_csv('table.csv')
# Sort by 'points' in descending order and get the top 5
top_5_points = df.sort_values(by='points', ascending=False).head(5)['points'].tolist()
highest_points = top_5_points[0]
fifth_highest_points = top_5_points[4]
difference = highest_points - fifth_highest_points
print(f"Final Answer: {difference}")