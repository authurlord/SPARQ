import pandas as pd

df = pd.read_csv('table.csv')
# Filter episodes that aired on Tuesdays
tuesday_episodes = df[df['timeslot'].str.contains('tuesday', case=False)]
# Calculate the average rating of those episodes
avg_rating_tuesday = tuesday_episodes['rating'].mean()
print(f"Final Answer: {avg_rating_tuesday:.1f}")