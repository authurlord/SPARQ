import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'viewers' column to numeric
df['viewers'] = pd.to_numeric(df['viewers'], errors='coerce')
# Find the episode with the highest viewership
max_viewers_episode = df.loc[df['viewers'].idxmax()]
print(f"Final Answer: {max_viewers_episode['title']}")