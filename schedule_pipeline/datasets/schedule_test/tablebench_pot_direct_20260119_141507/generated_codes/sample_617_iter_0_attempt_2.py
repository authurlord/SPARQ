import pandas as pd

df = pd.read_csv('table.csv')
# Filter seasons 1, 2, and 3, then sum the episodes
total_episodes = df[(df['Season'].isin([1, 2, 3]))]['Episodes'].sum()
print(f"Final Answer: {total_episodes}")