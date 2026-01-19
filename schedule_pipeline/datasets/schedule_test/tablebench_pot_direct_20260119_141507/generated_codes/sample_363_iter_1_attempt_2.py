import pandas as pd

df = pd.read_csv('table.csv')
# Convert the '2008 gdp per capita (usd) a' column to numeric, coercing errors to NaN
df['2008 gdp per capita (usd) a'] = pd.to_numeric(df['2008 gdp per capita (usd) a'], errors='coerce')

# Filter districts with GDP per capita above 6700 and count them
count_above_6700 = df[df['2008 gdp per capita (usd) a'] > 6700].shape[0]
print(f"Final Answer: {count_above_6700}")