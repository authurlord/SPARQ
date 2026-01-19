import pandas as pd

df = pd.read_csv('table.csv')

# Filter for first 10 rounds
df_first_10 = df[df['round'].astype(int) <= 10]

# Total number of players drafted in first 10 rounds
total_players = len(df_first_10)

# Count players per college in first 10 rounds
college_counts = df_first_10['college'].value_counts()

# Calculate pick efficiency for each college
pick_efficiency = college_counts / total_players

# Find the college with the highest pick efficiency
best_college = pick_efficiency.idxmax()

print(f"Final Answer: {best_college}")