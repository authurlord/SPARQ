import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'rating' column to float
df['rating'] = pd.to_numeric(df['rating'])
# Find the episode with the lowest rating
lowest_rating_episode = df.loc[df['rating'].idxmin(), 'episode']
print(f"Final Answer: {lowest_rating_episode}")