import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'reg gp' to integer type to avoid conversion errors
df['reg gp'] = pd.to_numeric(df['reg gp'], errors='coerce')

# Find Harold Snepsts' regular season games
harold_games = df[df['player'] == 'harold snepsts']['reg gp'].values[0]

# Average number of regular season games
average_games = 300

# Calculate how many more games Harold played than the average
difference = harold_games - average_games
print(f"Final Answer: {difference}")