import pandas as pd

df = pd.read_csv('table.csv')
# Filter for gold medals (Position = '1st')
gold_medals = df[df['Position'] == '1st']
# Count gold medals per year
gold_counts = gold_medals['Year'].value_counts()
# Find the year with the most gold medals
max_gold_year = gold_counts.idxmax()
print(f"Final Answer: {max_gold_year}")