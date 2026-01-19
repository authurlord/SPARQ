import pandas as pd

df = pd.read_csv('table.csv')

# Filter for first 10 rounds
first_10_rounds = df[df['round'].astype(int) <= 10]

# Total number of players drafted in first 10 rounds
total_players = len(first_10_rounds)

# Count number of players per college in first 10 rounds
college_counts = first_10_rounds['college'].value_counts()

# Calculate pick efficiency for each college
pick_efficiency = college_counts / total_players

# Find the college with the highest pick efficiency
best_college = pick_efficiency.idxmax()

print(f"Final Answer: {best_college}")