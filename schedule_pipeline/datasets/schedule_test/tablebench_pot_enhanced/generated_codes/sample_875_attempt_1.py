import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'points' column to integer
df['points'] = pd.to_numeric(df['points'])
# Sort by points in descending order and get top two
top_two_points = df['points'].sort_values(ascending=False).head(2)
# Calculate the difference
difference = top_two_points.iloc[0] - top_two_points.iloc[1]
print(f"Final Answer: {difference}")