import pandas as pd

df = pd.read_csv('table.csv')
# Filter teams with win percentage greater than 0.7
filtered_df = df[df['win pct'].astype(float) > 0.7]
# Sum the wins for these teams
total_wins = filtered_df['wins'].astype(int).sum()
print(f"Final Answer: {total_wins}")