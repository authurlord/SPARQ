import pandas as pd

df = pd.read_csv('table.csv')
# Find the city with the lowest GDP per capita
lowest_gdp_city = df.loc[df['gdp per capita id k'].idxmin(), 'city']
print(f"Final Answer: {lowest_gdp_city}")