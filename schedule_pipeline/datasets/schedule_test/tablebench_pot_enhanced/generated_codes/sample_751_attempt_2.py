import pandas as pd

df = pd.read_csv('table.csv')
# Extract 'tower division' and 'year' data
tower_division = df[['year', 'tower division']].astype(int)
# Calculate the growth rate (linear regression approach)
years = tower_division['year'].values
populations = tower_division['tower division'].values
# Compute the slope (average annual growth)
slope = (populations[-1] - populations[0]) / (years[-1] - years[0])
# Project for 1881
projected_population = populations[-1] + slope * (1881 - years[-1])
print(f"Final Answer: {int(projected_population)}")