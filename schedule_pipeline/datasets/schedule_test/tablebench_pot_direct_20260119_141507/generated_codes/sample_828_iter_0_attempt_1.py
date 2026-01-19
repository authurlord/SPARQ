import pandas as pd

df = pd.read_csv('table.csv')
# Sort by 'gdp world rank' ascending to get top 20 countries
top_20 = df.sort_values(by='gdp world rank').head(20)
# Extract GDP per capita and compute median
median_gdp = top_20['gdp per capita'].median()
print(f"Final Answer: {median_gdp}")