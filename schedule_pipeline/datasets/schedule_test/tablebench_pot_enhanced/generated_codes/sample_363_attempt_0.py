import pandas as pd

df = pd.read_csv('table.csv')
# Convert '2008 gdp per capita (usd) a' to numeric
df['2008 gdp per capita (usd) a'] = pd.to_numeric(df['2008 gdp per capita (usd) a'])
# Filter districts with GDP per capita > 6700
high_gdp_per_capita = df[df['2008 gdp per capita (usd) a'] > 6700]
# Count the number of such districts
count = len(high_gdp_per_capita)
print(f"Final Answer: {count}")