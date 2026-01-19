import pandas as pd

df = pd.read_csv('table.csv')
# Convert '2008 gdp per capita (usd) a' to numeric
df['2008 gdp per capita (usd) a'] = pd.to_numeric(df['2008 gdp per capita (usd) a'], errors='coerce')
# Count districts with GDP per capita > 6700
count = df[df['2008 gdp per capita (usd) a'] > 6700].shape[0]
print(f"Final Answer: {count}")