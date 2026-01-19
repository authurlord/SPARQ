import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'points' column to integer
df['points'] = df['points'].astype(int)
# Sort by points in descending order
df_sorted = df.sort_values(by='points', ascending=False)
# Get the highest points (1st) and 5th highest points
highest_points = df_sorted.iloc[0]['points']
fifth_highest_points = df_sorted.iloc[4]['points']
# Calculate the difference
difference = highest_points - fifth_highest_points
print(f"Final Answer: {difference}")