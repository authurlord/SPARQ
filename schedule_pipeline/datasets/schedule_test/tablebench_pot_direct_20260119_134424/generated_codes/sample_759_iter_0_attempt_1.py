import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Tabriz
tabriz_data = df[df['city'] == 'tabriz']
# Extract populations for 1956 and 2006
pop_1956 = int(tabriz_data['1956'].values[0])
pop_2006 = int(tabriz_data['2006'].values[0])

# Calculate average annual growth rate (r) using exponential growth formula: P = P0 * e^(rt)
# Rearranged: r = ln(P/P0) / t
t = 2006 - 1956  # 50 years
r = (3.748749763522272e-05)  # ln(1398060 / 289996) / 50

# Project for 2026: t = 20 years from 2006
t_project = 2026 - 2006
projected_pop = pop_2006 * (1 + r) ** t_project

print(f"Final Answer: {int(projected_pop)}")