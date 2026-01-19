import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 100 m hurdles and gold medal (Position = '1st')
gold_medal_rows = df[(df['Event'] == '100 m hurdles') & (df['Position'] == '1st')]
# Get the first year they won a gold medal
first_gold_year = gold_medal_rows['Year'].iloc[0]
print(f"Final Answer: {first_gold_year}")