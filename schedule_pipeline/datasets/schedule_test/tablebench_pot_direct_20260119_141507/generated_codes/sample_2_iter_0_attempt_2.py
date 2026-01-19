import pandas as pd

df = pd.read_csv('table.csv')
# Filter countries with UN budget greater than 2%
filtered_df = df[df['un budget'] > '2.000%']
# Sum the GDP (nominal) of filtered countries
total_gdp_nominal = filtered_df['gdp (nominal) (millions of usd) 2011'].sum()
print(f"Final Answer: {total_gdp_nominal}")