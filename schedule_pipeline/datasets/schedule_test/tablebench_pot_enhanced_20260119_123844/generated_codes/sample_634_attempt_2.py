import pandas as pd

df = pd.read_csv('table.csv')
# Filter episodes that aired on Tuesdays
tuesday_episodes = df[df['timeslot'].str.contains('tuesday', case=False, na=False)]
# Calculate average rating for these episodes
avg_rating_tuesday = tuesday_episodes['rating'].astype(float).mean()
print(f"Final Answer: {avg_rating_tuesday:.2f}")