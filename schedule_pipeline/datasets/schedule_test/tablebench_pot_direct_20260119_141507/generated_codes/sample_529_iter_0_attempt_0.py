import pandas as pd

df = pd.read_csv('table.csv')
# Find the episode with the highest audience share (share (%))
max_share_row = df.loc[df['share (%)'].idxmax()]
episode_with_highest_share = max_share_row['episode']
print(f"Final Answer: {episode_with_highest_share}")