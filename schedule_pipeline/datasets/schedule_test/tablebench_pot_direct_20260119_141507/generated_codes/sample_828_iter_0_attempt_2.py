import pandas as pd

df = pd.read_csv('table.csv')
# Select the top 20 countries by world rank (first 20 rows)
top_20 = df.head(20)
# Extract GDP per capita values
gdp_per_capita = top_20['gdp per capita']
# Calculate the median
median_gdp = gdp_per_capita.median()
print(f"Final Answer: {median_gdp}")