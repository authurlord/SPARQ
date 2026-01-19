import pandas as pd

df = pd.read_csv('table.csv')
# Extract Tabriz row (index 3)
tabriz_data = df.iloc[3]
# Get population values from 1956 to 2006
pop_1956 = tabriz_data['1956']
pop_2006 = tabriz_data['2006']

# Calculate annual growth rate
n_years = 50  # from 1956 to 2006
growth_rate = (pop_2006 / pop_1956) ** (1 / n_years) - 1

# Project to 2026 (20 years after 2006)
years_to_project = 20
projected_pop = pop_2006 * (1 + growth_rate) ** years_to_project

print(f"Final Answer: {projected_pop:.0f}")