import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 100 m hurdles with 1st position
gold_medals = df[(df['Event'] == '100 m hurdles') & (df['Position'] == '1st')]
# Get the first year (earliest)
first_gold_year = gold_medals.iloc[0]['Year']
print(f"Final Answer: {first_gold_year}")