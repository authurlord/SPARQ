import pandas as pd

df = pd.read_csv('table.csv')
# Find the city with the minimum GDP per capita
min_gdp_per_capita_city = df.loc[df['gdp per capita id k'].idxmin(), 'city']
print(f"Final Answer: {min_gdp_per_capita_city}")