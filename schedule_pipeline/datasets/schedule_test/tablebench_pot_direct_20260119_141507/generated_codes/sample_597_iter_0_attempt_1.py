import pandas as pd

df = pd.read_csv('table.csv')
# Filter the first 5 episodes (no in series from 25 to 30)
first_5_episodes = df[df['no in series'].between(25, 30)]
# Calculate the average viewers for these episodes
average_viewers = first_5_episodes['viewers (millions)'].mean()
print(f"Final Answer: {average_viewers:.2f}")