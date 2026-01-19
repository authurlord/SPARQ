import pandas as pd

df = pd.read_csv('table.csv')
# Filter episodes with rating/share >= 3.0 and rank (timeslot) = 1
filtered_episodes = df[(df['rating / share (1849)'] >= '3.0') & (df['rank (timeslot)'] == 1)]
# Sum the viewers (in millions)
total_viewers = filtered_episodes['viewers (millions)'].sum()
print(f"Final Answer: {total_viewers}")