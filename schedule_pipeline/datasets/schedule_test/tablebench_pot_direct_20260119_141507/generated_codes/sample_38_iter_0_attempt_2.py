import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where season is 1
season_1_viewers = df[df['season'] == 1]['us viewers (million)']
# Calculate the mean
average_viewers = season_1_viewers.mean()
print(f"Final Answer: {average_viewers:.2f}")