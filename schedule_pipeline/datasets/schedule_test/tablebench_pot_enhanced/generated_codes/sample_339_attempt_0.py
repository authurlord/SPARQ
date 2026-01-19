import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 100 m hurdles and first position (gold medal)
gold_medal_row = df[(df['Event'] == '100 m hurdles') & (df['Position'] == '1st')]
# Get the first occurrence (earliest year)
first_gold_year = gold_medal_row['Year'].iloc[0]
print(f"Final Answer: {first_gold_year}")