import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'us viewers (millions)' to float for proper comparison
df['us viewers (millions)'] = pd.to_numeric(df['us viewers (millions)'])
# Find the episode with the lowest viewership
lowest_viewers_episode = df.loc[df['us viewers (millions)'].idxmin()]
print(f"Final Answer: {lowest_viewers_episode['title']}")