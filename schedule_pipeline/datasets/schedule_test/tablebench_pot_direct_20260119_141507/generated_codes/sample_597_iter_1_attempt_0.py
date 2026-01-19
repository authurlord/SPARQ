import pandas as pd

df = pd.read_csv('table.csv')
# Select the first 5 episodes based on 'no in series' (which is already in order)
first_5_episodes = df.head(5)
# Calculate the average viewers for these episodes
average_viewers = first_5_episodes['viewers (millions)'].mean()
print(f"Final Answer: {average_viewers:.2f}")