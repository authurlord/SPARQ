import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'points' column to integer
df['points'] = pd.to_numeric(df['points'])
# Sort by points in descending order and get top two scores
top_two = df.sort_values(by='points', ascending=False)['points'].head(2)
# Calculate the difference
difference = top_two.iloc[0] - top_two.iloc[1]
print(f"Final Answer: {difference}")