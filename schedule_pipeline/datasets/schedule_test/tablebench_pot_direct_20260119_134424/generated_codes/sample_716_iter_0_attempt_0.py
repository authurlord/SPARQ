import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'viewers' column to integer
df['viewers'] = df['viewers'].astype(int)
# Find the episode with the highest viewers
max_viewers_episode = df.loc[df['viewers'].idxmax()]
print(f"Final Answer: {max_viewers_episode['title']}")