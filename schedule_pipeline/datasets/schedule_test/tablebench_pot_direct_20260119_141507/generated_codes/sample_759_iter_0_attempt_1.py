import pandas as pd

df = pd.read_csv('table.csv')

# Filter row for Tabriz (rank 4)
tabriz_row = df[df['city'] == 'tabriz'].iloc[0]

# Extract population values from 1956 to 2006
years = ['1956', '1966', '1976', '1986', '1996', '2006']
populations = [tabriz_row[year] for year in years]

# Population in 1956 and 2006
p_1956 = populations[0]
p_2006 = populations[5]

# Calculate annual growth rate
growth_rate = (p_2006 / p_1956) ** (1/50)

# Project population in 2026 (20 years after 2006)
p_2026 = p_2006 * (1 + growth_rate) ** 20

print(f"Final Answer: {p_2026:.0f}")