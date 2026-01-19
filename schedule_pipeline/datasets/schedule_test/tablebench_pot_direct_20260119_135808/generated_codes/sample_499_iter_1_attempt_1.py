import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Liverpool in the 1995-96 season
liverpool_95_96 = df[(df['Club'] == 'Liverpool') & (df['Season'] == '1995–96')]
# Extract League appearances and goals
appearances = int(liverpool_95_96['League'].values[0])
goals = int(liverpool_95_96['League_2'].values[0])
# Calculate goal percentage
goal_percentage = (goals / appearances) * 100
print(f"Final Answer: {goal_percentage:.2f}")