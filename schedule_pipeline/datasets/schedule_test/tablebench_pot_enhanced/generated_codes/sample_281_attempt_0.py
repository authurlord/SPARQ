import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'win pct' to float for comparison
df['win pct'] = pd.to_numeric(df['win pct'])
# Filter teams with win percentage > 0.7
filtered_teams = df[df['win pct'] > 0.7]
# Sum the 'wins' column for these teams
total_wins = filtered_teams['wins'].sum()
print(f"Final Answer: {total_wins}")