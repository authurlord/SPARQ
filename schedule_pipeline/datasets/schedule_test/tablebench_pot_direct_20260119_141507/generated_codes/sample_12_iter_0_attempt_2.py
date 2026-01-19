import pandas as pd

df = pd.read_csv('table.csv')
# Filter episodes with rating >= 6.0
filtered_episodes = df[df['rating'] >= '6.0']
# Calculate the average viewers (in millions)
average_viewers = filtered_episodes['viewers (millions)'].mean()
print(f"Final Answer: {average_viewers:.2f}")