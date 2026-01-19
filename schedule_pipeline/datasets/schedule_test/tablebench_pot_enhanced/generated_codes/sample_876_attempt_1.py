import pandas as pd

df = pd.read_csv('table.csv')
# Extract population density for 2012
pop_density_2012 = df['population density 2012 (km 2 )']
# Calculate the difference between highest and lowest density
density_difference = pop_density_2012.max() - pop_density_2012.min()
print(f"Final Answer: {density_difference}")