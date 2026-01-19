import pandas as pd

df = pd.read_csv('table.csv')
# Find the row where 'status' is 'champion , defeated rafael nadal'
champion_row = df[df['status'] == 'champion , defeated rafael nadal']
player_champion = champion_row['player'].iloc[0]
points_won = champion_row['points won'].iloc[0]
print(f"Final Answer: {player_champion}, {points_won}")