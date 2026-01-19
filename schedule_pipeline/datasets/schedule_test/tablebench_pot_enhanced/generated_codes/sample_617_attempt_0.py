import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for seasons 1, 2, and 3
filtered_df = df[df['Season'].isin(['1', '2', '3'])]
# Sum the episodes
total_episodes = filtered_df['Episodes'].sum()
print(f"Final Answer: {total_episodes}")