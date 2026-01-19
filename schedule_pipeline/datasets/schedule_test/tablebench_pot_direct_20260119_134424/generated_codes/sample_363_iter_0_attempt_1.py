import pandas as pd

df = pd.read_csv('table.csv')
# Convert '2008 gdp per capita (usd) a' to numeric
df['2008 gdp per capita (usd) a'] = pd.to_numeric(df['2008 gdp per capita (usd) a'], errors='coerce')
# Count districts with GDP per capita above 6700
count_above_6700 = (df['2008 gdp per capita (usd) a'] > 6700).sum()
print(f"Final Answer: {count_above_6700}")