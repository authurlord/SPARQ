import pandas as pd

df = pd.read_csv('table.csv')
# Find Harold Snepsts' regular season games
harold_snepsts_reg_gp = df[df['player'] == 'harold snepsts']['reg gp'].values[0]
# Calculate average regular season games
avg_reg_gp = df['reg gp'].mean()
# Compute difference
difference = harold_snepsts_reg_gp - avg_reg_gp
print(f"Final Answer: {difference}")