import pandas as pd

df = pd.read_csv('table.csv')
# Filter teams with win percentage >= 0.7
high_win_teams = df[df['win pct'] >= '0.7']
count = len(high_win_teams)
print(f"Final Answer: {count}")