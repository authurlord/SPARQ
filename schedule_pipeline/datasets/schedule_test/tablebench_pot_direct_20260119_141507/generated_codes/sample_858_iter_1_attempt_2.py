import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'gdp per capita (us)' to numeric, coercing errors to NaN
df['gdp per capita (us)'] = pd.to_numeric(df['gdp per capita (us)'], errors='coerce')

# Find the country with the highest GDP per capita
max_gdp_per_capita = df.loc[df['gdp per capita (us)'].idxmax(), 'member countries']
max_gdp_value = df['gdp per capita (us)'].max()

# GDP per capita of EC6 (1958)
ec6_gdp_per_capita = df.loc[df['member countries'] == 'ec6 (1958)', 'gdp per capita (us)'].values[0]

# Calculate the difference
difference = max_gdp_value - ec6_gdp_per_capita

print(f"Final Answer: {max_gdp_per_capita}, {difference:.0f}")