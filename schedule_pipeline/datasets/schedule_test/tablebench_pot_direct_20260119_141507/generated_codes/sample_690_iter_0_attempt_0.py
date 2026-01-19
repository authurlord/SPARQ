import pandas as pd

df = pd.read_csv('table.csv')
# Find the episode with the lowest rating
lowest_rating_episode = df.loc[df['rating'].idxmin(), 'episode']
print(f"Final Answer: {lowest_rating_episode}")