import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 100 m event and 1st position
gold_medal_year = df[(df['event'] == '100 m') & (df['position'] == '1st')]['year']
print(f"Final Answer: {gold_medal_year.values[0]}")