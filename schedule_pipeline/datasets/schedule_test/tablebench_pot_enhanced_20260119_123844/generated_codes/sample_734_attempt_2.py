import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'gdp per capita id k' to numeric for proper comparison
df['gdp per capita id k'] = pd.to_numeric(df['gdp per capita id k'])
# Find the city with the lowest GDP per capita
lowest_gdp_city = df.loc[df['gdp per capita id k'].idxmin(), 'city']
print(f"Final Answer: {lowest_gdp_city}")