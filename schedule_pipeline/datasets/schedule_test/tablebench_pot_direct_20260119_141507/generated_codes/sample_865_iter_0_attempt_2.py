import pandas as pd

df = pd.read_csv('table.csv')
# Extract total medals column
total_medals = df['total medals'].astype(int)
max_medals = total_medals.max()
min_medals = total_medals.min()
difference = max_medals - min_medals
print(f"Final Answer: {difference}")