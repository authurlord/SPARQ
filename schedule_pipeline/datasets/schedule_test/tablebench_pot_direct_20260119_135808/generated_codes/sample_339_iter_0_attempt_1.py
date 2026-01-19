import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 100 m hurdles and gold medal (position '1st')
gold_medals = df[(df['Event'] == '100 m hurdles') & (df['Position'] == '1st')]
# Get the earliest year
first_gold_year = gold_medals['Year'].min()
print(f"Final Answer: {first_gold_year}")