import pandas as pd

df = pd.read_csv('table.csv')
# Extract total medals column and find max and min
max_medals = df['total medals'].max()
min_medals = df['total medals'].min()
difference = max_medals - min_medals
print(f"Final Answer: {difference}")