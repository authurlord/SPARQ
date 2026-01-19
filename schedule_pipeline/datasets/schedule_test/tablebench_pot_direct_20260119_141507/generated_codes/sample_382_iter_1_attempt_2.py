import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'win pct' to float and filter out the totals row
df['win pct'] = df['win pct'].str.replace('%', '').astype(float)
# Remove the totals row
df_filtered = df[df['team'] != 'totals :']
# Count teams with win percentage >= 0.7
count_high_win_pct = df_filtered[df_filtered['win pct'] >= 0.7].shape[0]
print(f"Final Answer: {count_high_win_pct}")