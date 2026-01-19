import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for 2008 and 2009
wins_2008_2009 = df[df['year'].isin(['2008', '2009'])]['wins'].sum()
print(f"Final Answer: {wins_2008_2009}")