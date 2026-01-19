import pandas as pd

df = pd.read_csv('table.csv')
# Find the episode with the highest share percentage
max_share_episode = df.loc[df['share (%)'].idxmax(), 'episode']
print(f"Final Answer: {max_share_episode}")