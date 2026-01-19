import pandas as pd

df = pd.read_csv('table.csv')
# Filter players with total goals greater than 3
players_with_more_than_3_goals = df[df['total goals'] > 3]
# Count the number of such players
count = len(players_with_more_than_3_goals)
print(f"Final Answer: {count}")