import pandas as pd

df = pd.read_csv('table.csv')
# Filter episodes that aired on Tuesday
tuesday_episodes = df[df['timeslot'].str.contains('tuesday 9 / 8c', case=False)]
# Calculate the average rating of those episodes
average_rating = tuesday_episodes['rating'].mean()
print(f"Final Answer: {average_rating:.1f}")