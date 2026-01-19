import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where round is from 1 to 10 (inclusive)
df_filtered = df[df['round'].astype(int) <= 10]

# Group by college and count number of players
college_counts = df_filtered.groupby('college').size()

# Total number of players in rounds 1-10
total_players = len(df_filtered)

# Calculate efficiency: players from college / total players in the round range
efficiency = college_counts / total_players

# Find college with highest efficiency
max_efficiency_college = efficiency.idxmax()

print(f"Final Answer: {max_efficiency_college}")