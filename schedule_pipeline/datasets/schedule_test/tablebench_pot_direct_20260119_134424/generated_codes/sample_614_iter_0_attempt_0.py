import pandas as pd

df = pd.read_csv('table.csv')
# Find the 'reg gp' value for Harold Snepsts
harold_games = df[df['player'] == 'harold snepsts']['reg gp'].iloc[0]
# Calculate how many more games he played than the average
difference = int(harold_games) - 300
print(f"Final Answer: {difference}")