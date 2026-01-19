import pandas as pd

df = pd.read_csv('table.csv')
# Filter episodes that aired on Tuesdays
tuesday_episodes = df[df['timeslot'].str.contains('tuesday', case=False)]
# Convert rating to float and calculate mean
average_rating = tuesday_episodes['rating'].astype(float).mean()
print(f"Final Answer: {average_rating:.2f}")