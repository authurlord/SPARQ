import pandas as pd

df = pd.read_csv('table.csv')
# Count the number of gold medals (1st positions) per year
gold_medals_per_year = df[df['Position'] == '1st'].groupby('Year').size()
# Find the year with the maximum gold medals
max_gold_year = gold_medals_per_year.idxmax()
print(f"Final Answer: {max_gold_year}")