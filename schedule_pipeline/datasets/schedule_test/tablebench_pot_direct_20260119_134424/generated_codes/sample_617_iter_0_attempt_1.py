import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Seasons 1, 2, and 3
seasons_1_to_3 = df[df['Season'].isin(['1', '2', '3'])]
# Sum the episodes
total_episodes = seasons_1_to_3['Episodes'].sum()
print(f"Final Answer: {total_episodes}")