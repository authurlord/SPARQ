import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'win pct' to numeric, coercing errors to NaN if any
df['win pct'] = pd.to_numeric(df['win pct'], errors='coerce')

# Filter teams with win percentage > 0.7
filtered_teams = df[df['win pct'] > 0.7]

# Sum the wins of these teams
total_wins = filtered_teams['wins'].sum()
print(f"Final Answer: {total_wins}")