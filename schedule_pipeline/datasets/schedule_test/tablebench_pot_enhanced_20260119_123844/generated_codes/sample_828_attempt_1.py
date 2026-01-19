import pandas as pd

df = pd.read_csv('table.csv')
# Filter top 20 countries by world rank
top_20 = df.head(20)
# Convert 'gdp per capita' to numeric
top_20['gdp per capita'] = pd.to_numeric(top_20['gdp per capita'])
# Calculate median GDP per capita
median_gdp = top_20['gdp per capita'].median()
print(f"Final Answer: {median_gdp}")