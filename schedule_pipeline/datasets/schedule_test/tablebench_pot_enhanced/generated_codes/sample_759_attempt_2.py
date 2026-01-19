import pandas as pd

df = pd.read_csv('table.csv')

# Extract Tabriz's population data
tabriz_data = df[df['city'] == 'tabriz']
pop_1956 = int(tabriz_data['1956'].values[0])
pop_2006 = int(tabriz_data['2006'].values[0])

# Calculate average annual growth rate (CAGR)
years = 2006 - 1956
cagr = (pop_2006 / pop_1956) ** (1 / years) - 1

# Project population for 2026
years_to_project = 2026 - 2006
projected_pop = pop_2006 * (1 + cagr) ** years_to_project

print(f"Final Answer: {int(projected_pop)}")