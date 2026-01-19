import pandas as pd

df = pd.read_csv('table.csv')
# Extract population data for Tabriz (row with rank 4)
tabriz_data = df[df['rank'] == '4']
pop_1956 = int(tabriz_data['1956'].values[0])
pop_2006 = int(tabriz_data['2006'].values[0])

# Calculate average annual growth rate
years = 2006 - 1956
growth_rate = (pop_2006 / pop_1956) ** (1 / years) - 1

# Project population for 2026 (20 years after 2006)
projected_pop_2026 = pop_2006 * (1 + growth_rate) ** 20

print(f"Final Answer: {projected_pop_2026:.0f}")