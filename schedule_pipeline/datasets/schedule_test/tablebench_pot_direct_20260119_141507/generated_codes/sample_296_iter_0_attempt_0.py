import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Event is '200 m' and Competition is 'European Junior Championships' and Position is '1st'
gold_medal_200m = df[(df['Event'] == '200 m') & (df['Competition'] == 'European Junior Championships') & (df['Position'] == '1st')]
# Get the Year value
year = gold_medal_200m.iloc[0]['Year'] if not gold_medal_200m.empty else None
print(f"Final Answer: {year}")