import pandas as pd

df = pd.read_csv('table.csv')
# Filter episodes that aired on Tuesdays
tuesday_episodes = df[df['timeslot'].str.contains('tuesday', case=False, na=False)]
# Convert 'rating' to float and calculate average
average_rating_tuesday = tuesday_episodes['rating'].astype(float).mean()
print(f"Final Answer: {average_rating_tuesday:.2f}")