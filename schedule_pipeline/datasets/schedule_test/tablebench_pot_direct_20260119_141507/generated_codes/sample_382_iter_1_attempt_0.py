import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'win pct' to float and filter teams with win percentage >= 0.7
df['win pct'] = df['win pct'].str.replace('%', '').astype(float)
teams_high_win_pct = df[df['win pct'] >= 0.7]
count_high_win_pct = len(teams_high_win_pct)

print(f"Final Answer: {count_high_win_pct}")