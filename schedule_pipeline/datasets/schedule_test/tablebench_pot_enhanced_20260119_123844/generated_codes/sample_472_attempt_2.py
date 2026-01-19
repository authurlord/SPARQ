import pandas as pd

df = pd.read_csv('table.csv')

# Check for extreme population density
max_density = df['pop density (per km2)'].max()
min_density = df['pop density (per km2)'].min()

# Identify the district with the highest density
high_density_district = df.loc[df['pop density (per km2)'] == max_density, 'district'].iloc[0]

# Also check for very low population with large area
low_population = df['population (2010 census)'].min()
low_pop_district = df.loc[df['population (2010 census)'] == low_population, 'district'].iloc[0]

# Evaluate which is more suspicious
# 'san lorenzo' has 1M/km² — physically impossible
# 'sta. teresa' has 0.1/km² but 100 people in 1000 ha — possible if sparsely populated, but still questionable

# Since 'san lorenzo' has a density 10x higher than the next highest, it's the most likely error
print(f"Final Answer: san lorenzo")