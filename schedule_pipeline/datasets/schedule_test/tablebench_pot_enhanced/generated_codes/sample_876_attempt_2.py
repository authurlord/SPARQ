import pandas as pd

df = pd.read_csv('table.csv')
# Extract population density in 2012
population_density_2012 = df['population density 2012 (km 2 )']
# Calculate the difference between max and min density
density_difference = population_density_2012.max() - population_density_2012.min()
print(f"Final Answer: {density_difference}")