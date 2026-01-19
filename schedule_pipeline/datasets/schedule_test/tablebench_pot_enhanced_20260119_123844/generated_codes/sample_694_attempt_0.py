import pandas as pd

df = pd.read_csv('table.csv')
# Find the season with the maximum number of wins
max_wins_season = df.loc[df['wins'].idxmax(), 'season']
print(f"Final Answer: {max_wins_season}")