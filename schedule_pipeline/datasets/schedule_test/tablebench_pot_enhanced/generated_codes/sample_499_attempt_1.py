import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Liverpool in the 1995-96 season
liverpool_1995_96 = df[(df['Club'] == 'Liverpool') & (df['Season'] == '1995–96')]
# Extract apps and goals
apps = liverpool_1995_96['League'].iloc[0]  # League column contains Apps
goals = liverpool_1995_96['League_1'].iloc[0]  # League_1 column contains Goals
# Calculate goal percentage
goal_percentage = (goals / apps) * 100
print(f"Final Answer: {goal_percentage:.1f}")