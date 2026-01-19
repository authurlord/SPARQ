import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'win pct' to float and filter teams with win percentage > 0.7
df['win pct'] = pd.to_numeric(df['win pct'], errors='coerce')
filtered_teams = df[df['win pct'] > 0.7]
total_wins = filtered_teams['wins'].sum()
print(f"Final Answer: {total_wins}")