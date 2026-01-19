import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Liverpool in the 1995-96 season
liverpool_95_96 = df[(df['Club'] == 'Liverpool') & (df['Season'] == '1995–96')]
# Extract goals and appearances
goals = liverpool_95_96['League'].values[0]  # Goals from League column
apps = liverpool_95_96['League_1'].values[0]  # Apps from League_1 column
# Calculate goal percentage
goal_percentage = (goals / apps) * 100
print(f"Final Answer: {goal_percentage:.2f}")