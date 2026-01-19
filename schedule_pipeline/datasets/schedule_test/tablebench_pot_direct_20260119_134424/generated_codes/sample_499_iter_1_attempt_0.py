import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Liverpool in the 1995-1996 season
liverpool_1995_96 = df[(df['Club'] == 'Liverpool') & (df['Season'] == '1995–96')]
# Extract goals and appearances
goals = int(liverpool_1995_96['League'].iloc[0])
appearances = int(liverpool_1995_96['League_1'].iloc[0])
# Calculate goal percentage
goal_percentage = (goals / appearances) * 100
print(f"Final Answer: {goal_percentage:.2f}")