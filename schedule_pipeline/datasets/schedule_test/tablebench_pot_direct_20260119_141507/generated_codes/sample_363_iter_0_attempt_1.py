import pandas as pd

df = pd.read_csv('table.csv')
# Filter districts with 2008 GDP per capita above 6700 and count them
count_above_6700 = df[df['2008 gdp per capita (usd) a'] > 6700].shape[0]
print(f"Final Answer: {count_above_6700}")