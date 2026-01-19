import pandas as pd

df = pd.read_csv('table.csv')
# Sort by 'points' in descending order and get the highest and 5th highest
sorted_points = df.sort_values(by='points', ascending=False)
highest_points = sorted_points.iloc[0]['points']
fifth_highest_points = sorted_points.iloc[4]['points']  # 5th highest (index 4)

difference = highest_points - fifth_highest_points
print(f"Final Answer: {difference}")