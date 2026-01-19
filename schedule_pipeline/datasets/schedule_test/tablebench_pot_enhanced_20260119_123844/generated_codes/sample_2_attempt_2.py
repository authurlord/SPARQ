import pandas as pd

df = pd.read_csv('table.csv')
# Filter countries with UN budget greater than 2%
filtered_df = df[df['un budget'] > '2%']
# Convert GDP column to numeric and sum
total_gdp = filtered_df['gdp (nominal) (millions of usd) 2011'].astype(int).sum()
print(f"Final Answer: {total_gdp}")