import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Flevoland
flevoland_row = df[df['province'] == 'flevoland']
# Extract original GDP per capita
original_gdp_per_cap = flevoland_row['gdp per cap (2003 , in )'].values[0]
# Apply 37% decrease
new_gdp_per_cap = original_gdp_per_cap * (1 - 0.37)
print(f"Final Answer: {new_gdp_per_cap:.0f}")