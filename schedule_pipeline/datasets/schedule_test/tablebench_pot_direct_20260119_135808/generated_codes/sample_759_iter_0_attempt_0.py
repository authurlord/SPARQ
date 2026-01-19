import pandas as pd

df = pd.read_csv('table.csv')
# Filter row for Tabriz
tabriz_row = df[df['city'] == 'tabriz']
# Extract populations for 1956 and 2006
pop_1956 = int(tabriz_row['1956'].values[0])
pop_2006 = int(tabriz_row['2006'].values[0])

# Calculate average annual growth rate (r) using compound growth formula
# P = P0 * (1 + r)^t => r = (P/P0)^(1/t) - 1
t = 2006 - 1956  # 50 years
growth_rate = (pop_2006 / pop_1956) ** (1/t) - 1

# Project population for 2026 (20 years after 2006)
t_project = 2026 - 2006
projected_pop = pop_2006 * (1 + growth_rate) ** t_project

print(f"Final Answer: {int(projected_pop)}")