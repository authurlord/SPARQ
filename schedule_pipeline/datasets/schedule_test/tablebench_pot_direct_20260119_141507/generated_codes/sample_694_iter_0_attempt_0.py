import pandas as pd

df = pd.read_csv('table.csv')
# Group by 'season' and sum the 'wins' column
wins_by_season = df.groupby('season')['wins'].sum()
# Find the season with the highest number of wins
max_wins_season = wins_by_season.idxmax()
print(f"Final Answer: {max_wins_season}")