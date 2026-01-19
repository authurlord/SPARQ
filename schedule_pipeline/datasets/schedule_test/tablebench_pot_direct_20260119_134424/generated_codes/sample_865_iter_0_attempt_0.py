import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total medals' to integer
df['total medals'] = df['total medals'].astype(int)
# Find max and min total medals
max_medals = df['total medals'].max()
min_medals = df['total medals'].min()
# Calculate difference
difference = max_medals - min_medals
print(f"Final Answer: {difference}")