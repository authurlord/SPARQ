import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where status is 'champion'
champion_row = df[df['status'] == 'champion , defeated rafael nadal']
# Extract player and points won
player = champion_row['player'].iloc[0]
points_won = champion_row['points won'].iloc[0]
print(f"Final Answer: {player}, {points_won}")