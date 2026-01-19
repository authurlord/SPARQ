import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where event is '100 m' and position is '1st'
gold_medal_years = df[(df['event'] == '100 m') & (df['position'] == '1st')]['year']
# Since there's only one such year, return it
print(f"Final Answer: {gold_medal_years.iloc[0]}")