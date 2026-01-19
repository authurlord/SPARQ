import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'gdp per capita (us)' to numeric
df['gdp per capita (us)'] = pd.to_numeric(df['gdp per capita (us)'])

# Find the country with the highest GDP per capita
max_gdp_per_capita_country = df.loc[df['gdp per capita (us)'].idxmax(), 'member countries']
max_gdp_per_capita_value = df['gdp per capita (us)'].max()

# Get EC6 (1958) GDP per capita
ec6_gdp_per_capita = df[df['member countries'] == 'ec6 (1958)']['gdp per capita (us)'].values[0]

# Calculate the difference
difference = max_gdp_per_capita_value - ec6_gdp_per_capita

print(f"Final Answer: {max_gdp_per_capita_country}, {difference:.0f}")