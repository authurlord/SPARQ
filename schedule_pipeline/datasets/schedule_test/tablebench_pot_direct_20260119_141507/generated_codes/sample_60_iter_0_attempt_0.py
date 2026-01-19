import pandas as pd

df = pd.read_csv('table.csv')
# Find the season where wins is 13
season_13_wins = df[df['wins'] == 13]['season'].values[0]
print(f"Final Answer: {season_13_wins}")