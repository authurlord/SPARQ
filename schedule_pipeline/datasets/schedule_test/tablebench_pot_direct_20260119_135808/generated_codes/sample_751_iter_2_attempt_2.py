import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Extract data for 'tower division' and 'year'
tower_division = df['tower division'].astype(int)
years = df['year'].astype(int)

# Calculate the average growth per year
growth_rate_per_year = np.diff(tower_division) / np.diff(years)
avg_growth_rate = np.mean(growth_rate_per_year)

# Project for 1881 (from 1871)
last_population = tower_division[-1]
projected_population_1881 = last_population + avg_growth_rate * (1881 - 1871)

print(f"Final Answer: {int(round(projected_population_1881))}")