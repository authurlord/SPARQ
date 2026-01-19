import pandas as pd

df = pd.read_csv('table.csv')
# Extract the population density in 2012
population_density_2012 = df['population density 2012 (km 2 )']
# Find the highest and lowest population densities
max_density = population_density_2012.max()
min_density = population_density_2012.min()
# Calculate the difference
difference = max_density - min_density
print(f"Final Answer: {difference}")