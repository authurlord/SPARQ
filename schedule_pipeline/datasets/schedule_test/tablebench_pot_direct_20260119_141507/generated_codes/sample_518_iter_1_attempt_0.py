import pandas as pd

df = pd.read_csv('table.csv')

# Convert the GDP (nominal) and population columns to numeric
df['gdp (nominal) (millions of usd) 2011'] = pd.to_numeric(df['gdp (nominal) (millions of usd) 2011'], errors='coerce')
df['population'] = pd.to_numeric(df['population'], errors='coerce')

# Calculate GDP per capita
df['gdp_per_capita'] = df['gdp (nominal) (millions of usd) 2011'] / df['population']

# Find the country with the highest GDP per capita
highest_gdp_per_capita_country = df.loc[df['gdp_per_capita'].idxmax(), 'country']

print(f"Final Answer: {highest_gdp_per_capita_country}")