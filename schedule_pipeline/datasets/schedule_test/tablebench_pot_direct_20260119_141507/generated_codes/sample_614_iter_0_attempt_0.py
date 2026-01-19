import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the average of 'reg gp'
avg_reg_gp = df['reg gp'].mean()
# Find Harold Snepsts' reg gp
harold_snepsts_games = df[df['player'] == 'harold snepsts']['reg gp'].values[0]
# Compute the difference
difference = harold_snepsts_games - avg_reg_gp
print(f"Final Answer: {difference}")