import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Flevoland
flevoland_row = df[df['province'] == 'flevoland']
# Extract GDP per capita and convert to float
gdp_per_capita = float(flevoland_row['gdp per cap (2003 , in )'].values[0])
# Apply 37% decrease
new_gdp_per_capita = gdp_per_capita * (1 - 0.37)
print(f"Final Answer: {new_gdp_per_capita:.0f}")