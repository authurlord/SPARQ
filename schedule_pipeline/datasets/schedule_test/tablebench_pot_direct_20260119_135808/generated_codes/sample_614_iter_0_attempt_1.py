import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for Harold Snepsts
harold_games = df[df['player'] == 'harold snepsts']['reg gp'].values[0]
# Calculate how many more games he played than the average
difference = int(harold_games) - 300
print(f"Final Answer: {difference}")