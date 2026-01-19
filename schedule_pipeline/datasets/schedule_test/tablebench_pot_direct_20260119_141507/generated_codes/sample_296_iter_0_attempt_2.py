import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Event is '200 m' and Position is '1st'
gold_medal_200m = df[(df['Event'] == '200 m') & (df['Position'] == '1st')]
# Extract the Year for that row
year = gold_medal_200m.iloc[0]['Year']
print(f"Final Answer: {year}")