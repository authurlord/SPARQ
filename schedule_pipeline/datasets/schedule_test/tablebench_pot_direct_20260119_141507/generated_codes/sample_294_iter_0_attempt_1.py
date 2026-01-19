import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where event is '100 m' and position is '1st'
gold_medal_years = df[(df['event'] == '100 m') & (df['position'] == '1st')]['year']
# Convert to list and print the result
print(f"Final Answer: {', '.join(gold_medal_years.astype(str))}")