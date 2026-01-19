import pandas as pd

df = pd.read_csv('table.csv')
# Filter row for Tabriz
tabriz_data = df[df['city'] == 'tabriz']
# Extract populations from 1956 to 2006
populations = tabriz_data[['1956', '1966', '1976', '1986', '1996', '2006']].values[0].astype(int)
years = [1956, 1966, 1976, 1986, 1996, 2006]
# Calculate average annual growth rate
initial_pop = populations[0]
final_pop = populations[-1]
num_years = years[-1] - years[0]
growth_rate = (final_pop / initial_pop) ** (1 / num_years) - 1
# Project population for 2026
projected_pop_2026 = final_pop * (1 + growth_rate) ** (2026 - 2006)
print(f"Final Answer: {int(projected_pop_2026)}")