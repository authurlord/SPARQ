import pandas as pd

df = pd.read_csv('table.csv')

# Extract Tabriz's population data
tabriz_data = df[df['city'] == 'tabriz']
populations = tabriz_data[['1956', '1966', '1976', '1986', '1996', '2006']].values[0]
years = [1956, 1966, 1976, 1986, 1996, 2006]

# Convert to integers
populations = [int(p) for p in populations]

# Calculate CAGR from 1956 to 2006 (50 years)
initial_pop = populations[0]
final_pop = populations[-1]
n = len(years) - 1  # 5 intervals of 10 years, total 50 years
cagr = (final_pop / initial_pop) ** (1 / n) - 1

# Project population for 2026 (20 years from 2006)
projected_pop_2026 = final_pop * (1 + cagr) ** (20 / 10)

# Round to nearest whole number
projected_pop_2026 = round(projected_pop_2026)

print(f"Final Answer: {projected_pop_2026}")