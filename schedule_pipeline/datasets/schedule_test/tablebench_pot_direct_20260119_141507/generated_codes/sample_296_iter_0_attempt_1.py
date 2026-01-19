import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Event is '200 m' and Position is '1st'
gold_medal_200m = df[(df['Event'] == '200 m') & (df['Position'] == '1st')]
# Extract the Year for such a row
year_gold_200m = gold_medal_200m.iloc[0]['Year'] if not gold_medal_200m.empty else None
print(f"Final Answer: {year_gold_200m}")