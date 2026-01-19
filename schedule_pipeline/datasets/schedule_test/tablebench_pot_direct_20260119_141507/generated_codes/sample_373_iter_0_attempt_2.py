import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where League is "Premier League"
premier_league_seasons = df[df['League'] == 'Premier League']
# Count the number of such seasons
num_seasons = len(premier_league_seasons)
print(f"Final Answer: {num_seasons}")