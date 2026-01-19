import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 100 m event and gold medal (position '1st')
gold_medal_year = df[(df['event'] == '100 m') & (df['position'] == '1st')]['year'].iloc[0]
print(f"Final Answer: {gold_medal_year}")