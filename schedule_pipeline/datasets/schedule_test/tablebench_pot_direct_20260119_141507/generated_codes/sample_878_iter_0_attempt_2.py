import pandas as pd

df = pd.read_csv('table.csv')
# Sort by 'points' in descending order and get the top 5
sorted_df = df.sort_values(by='points', ascending=False)
highest_points = sorted_df.iloc[0]['points']
fifth_highest_points = sorted_df.iloc[4]['points']
difference = highest_points - fifth_highest_points
print(f"Final Answer: {difference}")