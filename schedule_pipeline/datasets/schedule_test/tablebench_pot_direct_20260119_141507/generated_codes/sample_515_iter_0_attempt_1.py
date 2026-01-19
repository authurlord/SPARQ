import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows for rounds 1 to 10 (inclusive)
df_filtered = df[df['round'].astype(int) <= 10]

# Group by college and count number of players
college_counts = df_filtered['college'].value_counts()

# Total number of players drafted in rounds 1-10
total_players = len(df_filtered)

# Calculate efficiency for each college: count / total_players
efficiency = college_counts / total_players

# Find the college with the highest efficiency
max_efficiency_college = efficiency.idxmax()

print(f"Final Answer: {max_efficiency_college}")