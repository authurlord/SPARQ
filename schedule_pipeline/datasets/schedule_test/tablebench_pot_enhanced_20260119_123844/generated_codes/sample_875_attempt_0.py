import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'points' column to integer
df['points'] = pd.to_numeric(df['points'])
# Sort by points in descending order
df_sorted = df.sort_values(by='points', ascending=False)
# Get the top two scores
highest_points = df_sorted.iloc[0]['points']
second_highest_points = df_sorted.iloc[1]['points']
# Calculate the difference
difference = highest_points - second_highest_points
print(f"Final Answer: {difference}")