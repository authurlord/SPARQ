import pandas as pd

df = pd.read_csv('table.csv')

# Extract Tabriz's population data
tabriz_data = df[df['city'] == 'tabriz']
years = [1956, 1966, 1976, 1986, 1996, 2006]
populations = [int(tabriz_data[f'{year}'].values[0]) for year in years]

# Calculate average annual growth rate (CAGR)
start_pop = populations[0]
end_pop = populations[-1]
n = len(years) - 1  # Number of years between 1956 and 2006
cagr = (end_pop / start_pop) ** (1 / n) - 1

# Project population for 2026 (20 years after 2006)
projected_pop_2026 = end_pop * (1 + cagr) ** 20

# Round to nearest whole number
print(f"Final Answer: {round(projected_pop_2026)}")