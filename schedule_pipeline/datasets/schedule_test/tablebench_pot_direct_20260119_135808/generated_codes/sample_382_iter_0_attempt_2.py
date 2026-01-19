import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'win pct' to float and filter teams with win percentage >= 0.7
teams_with_high_win_pct = df[df['win pct'].astype(float) >= 0.7]
count = len(teams_with_high_win_pct)
print(f"Final Answer: {count}")