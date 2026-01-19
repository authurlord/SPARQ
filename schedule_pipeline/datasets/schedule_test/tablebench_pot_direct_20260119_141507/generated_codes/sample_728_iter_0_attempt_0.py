import pandas as pd

df = pd.read_csv('table.csv')
# Find the episode with the minimum US viewers
min_viewers_episode = df.loc[df['us viewers (millions)'].idxmin(), 'title']
print(f"Final Answer: {min_viewers_episode}")