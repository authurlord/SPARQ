import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'avg finish' to float
df['avg finish'] = pd.to_numeric(df['avg finish'], errors='coerce')
# Convert 'winnings' to float
df['winnings'] = pd.to_numeric(df['winnings'], errors='coerce')

# Calculate average finish across all seasons
avg_finish = df['avg finish'].mean()

# Calculate average winnings per race
avg_winnings_per_race = df['winnings'].mean() / df['starts'].mean()

# Projected winnings for 5 races (same as last active year)
projected_winnings = avg_winnings_per_race * 5

print(f"Final Answer: {avg_finish:.1f}, {projected_winnings:.0f}")