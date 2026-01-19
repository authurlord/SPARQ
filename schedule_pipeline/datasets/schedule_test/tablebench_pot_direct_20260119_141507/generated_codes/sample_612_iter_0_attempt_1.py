import pandas as pd

df = pd.read_csv('table.csv')
# Extract GDP per capita for Flevoland
gdp_per_capita_flevoland = df[df['province'] == 'flevoland']['gdp per cap (2003 , in)'].values[0]
# Calculate 37% decrease
new_gdp_per_capita = gdp_per_capita_flevoland * (1 - 0.37)
print(f"Final Answer: {new_gdp_per_capita:.0f}")