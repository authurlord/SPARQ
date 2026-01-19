import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'win pct' to float and filter teams with win percentage > 0.7
filtered_df = df[df['win pct'].astype(float) > 0.7]
# Sum the 'wins' column for these teams
total_wins = filtered_df['wins'].astype(int).sum()
print(f"Final Answer: {total_wins}")