import pandas as pd

df = pd.read_csv('table.csv')
# Find the row where 'status' is 'champion'
champion_row = df[df['status'] == 'champion']
# Extract player and points won
champion_player = champion_row['player'].iloc[0]
points_won = champion_row['points won'].iloc[0]
print(f"Final Answer: {champion_player}, {points_won}")