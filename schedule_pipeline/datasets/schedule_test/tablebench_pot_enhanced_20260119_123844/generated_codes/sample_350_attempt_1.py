import pandas as pd

df = pd.read_csv('table.csv')
# Filter players with total goals > 3
players_above_3_goals = df[df['total goals'] > 3]
count = len(players_above_3_goals)
print(f"Final Answer: {count}")