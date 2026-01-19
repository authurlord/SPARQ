import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Liverpool in the 1995-96 season
liverpool_9596 = df[(df['Club'] == 'Liverpool') & (df['Season'] == '1995–96')]
# Extract apps and goals from League column
apps = liverpool_9596['League'].values[0]
goals = liverpool_9596['League_1'].values[0]
# Calculate goal percentage
goal_percentage = (goals / apps) * 100
print(f"Final Answer: {goal_percentage:.2f}")