import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'gdp per capita (us)' to numeric
df['gdp per capita (us)'] = pd.to_numeric(df['gdp per capita (us)'])
# Find the country with the highest GDP per capita
max_gdp_per_capita_country = df.loc[df['gdp per capita (us)'].idxmax()]
# Get the GDP per capita of the country and EC6 (1958)
highest_gdp_per_capita = max_gdp_per_capita_country['gdp per capita (us)']
ec6_gdp_per_capita = df[df['member countries'] == 'ec6 (1958)']['gdp per capita (us)'].values[0]
# Calculate the difference
difference = highest_gdp_per_capita - ec6_gdp_per_capita
print(f"Final Answer: {max_gdp_per_capita_country['member countries']}, {difference}")