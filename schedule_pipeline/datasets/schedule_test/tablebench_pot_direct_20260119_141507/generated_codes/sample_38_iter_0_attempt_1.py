import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where season is 1
season_1_episodes = df[df['season'] == 1]
# Calculate the mean of 'us viewers (million)' for those episodes
average_viewers = season_1_episodes['us viewers (million)'].mean()
print(f"Final Answer: {average_viewers:.2f}")