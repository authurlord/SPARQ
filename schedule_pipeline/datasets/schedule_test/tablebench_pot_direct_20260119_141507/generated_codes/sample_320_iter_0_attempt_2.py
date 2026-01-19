import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Position is '1st' and Event is '20 km'
gold_medal_years = df[(df['Position'] == '1st') & (df['Event'] == '20 km')]['Year']
# Since the question asks for "in which year", we return the year(s) as a list
print(f"Final Answer: {list(gold_medal_years)}")