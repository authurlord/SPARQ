import pandas as pd

df = pd.read_csv('table.csv')
# Remove the "totals :" row which is not a team
df = df[df['team'] != 'totals :']

# Convert 'win pct' to float for numerical comparison
df['win pct'] = df['win pct'].str.replace('%', '').astype(float)

# Filter teams with win percentage > 0.7
high_win_pct_teams = df[df['win pct'] > 0.7]

# Sum the wins of these teams
total_wins = high_win_pct_teams['wins'].sum()
print(f"Final Answer: {total_wins}")