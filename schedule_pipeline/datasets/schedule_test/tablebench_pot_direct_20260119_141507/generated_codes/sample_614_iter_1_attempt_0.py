import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'reg gp' to integer type to avoid conversion errors
df['reg gp'] = pd.to_numeric(df['reg gp'], errors='coerce')

# Calculate average regular season games played
average_reg_gp = df['reg gp'].mean()

# Harold Snepsts' regular season games
harold_games = 781

# Difference between Harold Snepsts and average
difference = harold_games - average_reg_gp

print(f"Final Answer: {difference:.0f}")