import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract Tabriz data
tabriz_data = df[df['city'] == 'tabriz']

# Extract population values from 1956 to 2006
pop_1956 = int(tabriz_data.iloc[0][5])  # 1956 column
pop_2006 = int(tabriz_data.iloc[0][9])  # 2006 column

# Calculate annual growth rate (exponential)
years = 2006 - 1956
growth_rate = (pop_2006 / pop_1956) ** (1 / years) - 1

# Project to 2026 (20 years after 2006)
years_to_project = 2026 - 2006
projected_population = pop_2006 * (1 + growth_rate) ** years_to_project

print(f"Final Answer: {int(projected_population):,}")