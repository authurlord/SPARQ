import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Harold Snepsts and get his reg gp
harold_games = df[df['player'] == 'harold snepsts']['reg gp'].values[0]
# Calculate difference from average
difference = int(harold_games) - 300
print(f"Final Answer: {difference}")