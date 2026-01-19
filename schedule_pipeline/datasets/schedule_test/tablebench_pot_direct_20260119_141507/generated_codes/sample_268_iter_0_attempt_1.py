import pandas as pd

df = pd.read_csv('table.csv')
# Filter episodes with viewership >= 10 million and timeslot rank <= 3
filtered_episodes = df[(df['viewers (m)'] >= 10) & (df['timeslot rank'] <= 3)]
# Calculate the average rating of the filtered episodes
average_rating = filtered_episodes['rating'].mean()
print(f"Final Answer: {average_rating:.1f}")