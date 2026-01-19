import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the "totals :" row and select only teams with win percentage >= 0.7
teams = df[df['win pct'] >= 0.7]
# Exclude the last row (totals) which is not a team
teams = teams[teams['team'] != 'totals :']
# Count the number of such teams
count = len(teams)
print(f"Final Answer: {count}")