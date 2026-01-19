import pandas as pd

df = pd.read_csv('table.csv')
# Find Harold Snepsts' regular season games
harold_games = df[df['player'] == 'harold snepsts']['reg gp'].values[0]
average_games = 300
difference = harold_games - average_games
print(f"Final Answer: {difference}")