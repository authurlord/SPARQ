import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'us viewers (millions)' to float for numerical comparison
df['us viewers (millions)'] = pd.to_numeric(df['us viewers (millions)'], errors='coerce')
# Find the episode with the lowest US viewers
lowest_viewer_episode = df.loc[df['us viewers (millions)'].idxmin()]
print(f"Final Answer: {lowest_viewer_episode['title']}")