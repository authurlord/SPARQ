import pandas as pd

df = pd.read_csv('table.csv')
# Find the player with the maximum total goals
max_goals_player = df.loc[df['total goals'].idxmax(), 'name']
print(f"Final Answer: {max_goals_player}")