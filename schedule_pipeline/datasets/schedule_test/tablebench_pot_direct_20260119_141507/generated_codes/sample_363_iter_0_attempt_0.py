import pandas as pd

df = pd.read_csv('table.csv')
# Filter districts with 2008 GDP per capita above 6700
filtered = df[df['2008 gdp per capita (usd) a'] > 6700]
count = len(filtered)
print(f"Final Answer: {count}")