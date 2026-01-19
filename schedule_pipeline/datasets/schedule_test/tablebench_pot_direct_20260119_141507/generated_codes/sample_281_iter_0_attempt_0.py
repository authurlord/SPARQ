import pandas as pd

df = pd.read_csv('table.csv')
# Filter teams with win percentage greater than 0.7
filtered_df = df[df['win pct'] > 0.7]
# Sum the wins of these teams
total_wins = filtered_df['wins'].sum()
print(f"Final Answer: {total_wins}")