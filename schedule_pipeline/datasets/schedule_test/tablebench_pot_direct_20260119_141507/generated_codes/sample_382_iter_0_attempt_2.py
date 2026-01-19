import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the "totals :" row and select only teams (not the total row)
df_teams = df[df['team'] != 'totals :']
# Filter teams with win percentage >= 0.7
high_win_pct_teams = df_teams[df_teams['win pct'] >= 0.7]
# Count the number of such teams
count = len(high_win_pct_teams)
print(f"Final Answer: {count}")