import pandas as pd

df = pd.read_csv('table.csv')

# Filter players by position
forwards = df[df['position'] == 'fw']
midfielders = df[df['position'] == 'mf']

# Find the top scorer in each group by total goals
top_forward = forwards.loc[forwards['total goals'].idxmax()]
top_midfielder = midfielders.loc[midfielders['total goals'].idxmax()]

# Calculate the difference in total goals
goal_difference = top_forward['total goals'] - top_midfielder['total goals']
print(f"Final Answer: {goal_difference}")