import pandas as pd

df = pd.read_csv('table.csv')
# Exclude the "totals :" row
df_filtered = df[df['team'] != 'totals :']
# Filter teams with win percentage >= 0.7
high_win_pct_teams = df_filtered[df_filtered['win pct'] >= 0.7]
count = len(high_win_pct_teams)
print(f"Final Answer: {count}")