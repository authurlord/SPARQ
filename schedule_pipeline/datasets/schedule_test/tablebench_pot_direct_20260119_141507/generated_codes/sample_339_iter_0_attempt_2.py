import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 1st place in 100 m hurdles
gold_medal_year = df[(df['Position'] == '1st') & (df['Event'] == '100 m hurdles')]['Year'].iloc[0]
print(f"Final Answer: {gold_medal_year}")