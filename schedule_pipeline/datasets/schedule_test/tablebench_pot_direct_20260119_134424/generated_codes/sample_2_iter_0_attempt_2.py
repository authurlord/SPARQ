import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'un budget' to float by removing the % sign and dividing by 100
df['un budget'] = df['un budget'].str.replace('%', '').astype(float)
# Filter countries with UN budget > 2%
filtered_df = df[df['un budget'] > 2]
# Sum the GDP (nominal) for these countries
total_gdp_nominal = filtered_df['gdp (nominal) (millions of usd) 2011'].sum()
print(f"Final Answer: {total_gdp_nominal}")