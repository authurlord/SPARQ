import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Club is 'Liverpool' and League is 'Premier League'
premier_league_seasons = df[(df['Club'] == 'Liverpool') & (df['League'] == 'Premier League')]
# Exclude the total row (last row for Liverpool)
premier_league_seasons = premier_league_seasons[~(premier_league_seasons['Season'] == 'Liverpool Total')]
# Count the number of such rows
num_seasons = len(premier_league_seasons)
print(f"Final Answer: {num_seasons}")