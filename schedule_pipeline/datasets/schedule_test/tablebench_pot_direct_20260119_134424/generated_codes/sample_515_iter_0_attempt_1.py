import pandas as pd

df = pd.read_csv('table.csv')
# Filter for rounds 1 to 10
df_rounds_1_to_10 = df[df['round'].astype(int) <= 10]

# Total number of players drafted in rounds 1 to 10
total_players = len(df_rounds_1_to_10)

# Count number of players drafted from each college
college_counts = df_rounds_1_to_10['college'].value_counts()

# Calculate pick efficiency for each college
pick_efficiency = college_counts / total_players

# Find the college with the highest pick efficiency
best_college = pick_efficiency.idxmax()

print(f"Final Answer: {best_college}")