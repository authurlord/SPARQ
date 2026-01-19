import pandas as pd

df = pd.read_csv('table.csv')

# Extract Tabriz's population data
tabriz_data = df[df['city'] == 'tabriz']
populations = [int(tabriz_data[f'{year}'].values[0]) for year in [1956, 1966, 1976, 1986, 1996, 2006]]
years = [1956, 1966, 1976, 1986, 1996, 2006]

# Calculate average annual growth rate
total_growth_rate = (populations[-1] / populations[0]) ** (1 / len(years)) - 1
# Project population for 2026 (20 years after 2006)
projected_pop_2026 = populations[-1] * (1 + total_growth_rate) ** 20

print(f"Final Answer: {int(projected_pop_2026)}")