import pandas as pd

df = pd.read_csv('table.csv')
# Convert '2008 gdp per capita (usd) a' to numeric, handling any potential non-numeric values
df['2008 gdp per capita (usd) a'] = pd.to_numeric(df['2008 gdp per capita (usd) a'], errors='coerce')
# Count districts with GDP per capita > 6700
count = (df['2008 gdp per capita (usd) a'] > 6700).sum()
print(f"Final Answer: {count}")