import pandas as pd

df = pd.read_csv('table.csv')
# Calculate GDP per capita
df['gdp_per_capita'] = df['gdp (nominal) (millions of usd) 2011'] / df['population']
# Find the country with the highest GDP per capita
highest_gdp_per_capita_country = df.loc[df['gdp_per_capita'].idxmax(), 'country']
print(f"Final Answer: {highest_gdp_per_capita_country}")