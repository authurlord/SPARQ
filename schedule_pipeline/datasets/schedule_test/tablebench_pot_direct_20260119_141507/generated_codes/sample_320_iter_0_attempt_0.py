import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where position is '1st' and event is '20 km'
gold_medal_years = df[(df['Position'] == '1st') & (df['Event'] == '20 km')]['Year']
# Since there might be multiple years, we return all of them
print(f"Final Answer: {', '.join(gold_medal_years.tolist())}")