import pandas as pd

df = pd.read_csv('table.csv')
# Filter countries with UN budget > 2%
filtered_df = df[df['un budget'] > '2%']
# Convert GDP (nominal) to numeric and sum
total_gdp_nominal = filtered_df['gdp (nominal) (millions of usd) 2011'].astype(float).sum()
print(f"Final Answer: {total_gdp_nominal}")