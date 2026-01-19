import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where status is 'champion'
champion_row = df[df['status'].str.contains('champion', case=False, na=False)]
# Extract player and points won
player = champion_row['player'].values[0]
points_won = champion_row['points won'].values[0]
print(f"Final Answer: {player}, {points_won}")