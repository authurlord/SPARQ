import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'win pct' to float and filter teams with win percentage >= 0.7
df['win pct'] = pd.to_numeric(df['win pct'], errors='coerce')
teams_high_win_pct = df[df['win pct'] >= 0.7]
count_teams = len(teams_high_win_pct)
print(f"Final Answer: {count_teams}")