import pandas as pd

df = pd.read_csv('table.csv')
# Filter episodes for Seasons 1, 2, and 3
filtered_episodes = df[(df['Season'].isin([1, 2, 3]))]['Episodes']
total_episodes = filtered_episodes.sum()
print(f"Final Answer: {total_episodes}")