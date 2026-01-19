import pandas as pd

df = pd.read_csv('table.csv')
# Group by 'season' and sum the 'wins' column
wins_per_season = df.groupby('season')['wins'].sum()
# Find the season with the highest total wins
max_wins_season = wins_per_season.idxmax()
print(f"Final Answer: {max_wins_season}")