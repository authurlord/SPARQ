import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 100 m hurdles and 1st position
gold_medal_rows = df[(df['Event'] == '100 m hurdles') & (df['Position'] == '1st')]
# Get the first year (earliest) when a gold medal was won
first_gold_year = gold_medal_rows['Year'].iloc[0]
print(f"Final Answer: {first_gold_year}")