import pandas as pd

df = pd.read_csv('table.csv')
# Filter row for Tabriz
tabriz_row = df[df['city'] == 'tabriz']
# Extract populations in 1956 and 2006
pop_1956 = int(tabriz_row['1956'].values[0])
pop_2006 = int(tabriz_row['2006'].values[0])

# Calculate average annual growth rate
years = 2006 - 1956
growth_rate = (pop_2006 / pop_1956) ** (1 / years) - 1

# Project population for 2026 (20 years from 2006)
projected_pop_2026 = pop_2006 * (1 + growth_rate) ** 20

print(f"Final Answer: {int(projected_pop_2026)}")