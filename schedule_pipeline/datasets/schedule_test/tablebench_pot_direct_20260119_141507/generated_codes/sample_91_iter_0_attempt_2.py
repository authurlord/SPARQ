import pandas as pd

df = pd.read_csv('table.csv')
# Find the team with the minimum points
least_points_team = df.loc[df['points'].idxmin(), 'team']
print(f"Final Answer: {least_points_team}")