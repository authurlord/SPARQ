import pandas as pd

df = pd.read_csv('table.csv')
# Extract viewership data
viewers = df['viewers (in millions)'].astype(float)
# Calculate the average change per episode
avg_change = viewers.diff().mean()
# Predict next episode viewership
next_episode_viewers = viewers.iloc[-1] + avg_change
print(f"Final Answer: {next_episode_viewers:.2f}")