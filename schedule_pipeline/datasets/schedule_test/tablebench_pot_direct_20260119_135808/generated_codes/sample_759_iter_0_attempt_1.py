import pandas as pd

df = pd.read_csv('table.csv')

# Extract Tabriz's population data
tabriz_data = df[df['city'] == 'tabriz']
populations = [int(tabriz_data.iloc[0][f'{year}']) for year in [1956, 1966, 1976, 1986, 1996, 2006]]

# Calculate average annual growth rate
initial_pop = populations[0]
final_pop = populations[-1]
years = len(populations) - 1  # 5 decades = 50 years
growth_rate = (final_pop / initial_pop) ** (1 / years) - 1

# Project population for 2026 (20 years after 2006)
projected_pop_2026 = final_pop * (1 + growth_rate) ** 20

print(f"Final Answer: {projected_pop_2026:.0f}")