import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'gdp per capita (us)' to numeric, coercing errors to NaN if any
df['gdp per capita (us)'] = pd.to_numeric(df['gdp per capita (us)'], errors='coerce')

# Find the country with the highest GDP per capita
max_gdp_per_capita = df['gdp per capita (us)'].max()
max_gdp_country = df[df['gdp per capita (us)'] == max_gdp_per_capita]['member countries'].values[0]

# Get GDP per capita of EC6 (1958)
ec6_gdp_per_capita = df[df['member countries'] == 'ec6 (1958)']['gdp per capita (us)'].values[0]

# Calculate the difference
difference = max_gdp_per_capita - ec6_gdp_per_capita

print(f"Final Answer: {max_gdp_country}, {difference:.0f}")