import pandas as pd

df = pd.read_csv('table.csv')
# Ensure 'reg gp' is numeric
df['reg gp'] = pd.to_numeric(df['reg gp'], errors='coerce')

# Find Harold Snepsts' regular season games
harold_snepsts_games = df[df['player'] == 'harold snepsts']['reg gp'].values[0]

# Average given in the question
average_games = 300

# Calculate difference
difference = harold_snepsts_games - average_games
print(f"Final Answer: {difference}")