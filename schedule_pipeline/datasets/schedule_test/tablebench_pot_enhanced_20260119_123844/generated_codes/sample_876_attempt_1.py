import pandas as pd

df = pd.read_csv('table.csv')
# Extract population density for 2012
density_2012 = df['population density 2012 (km 2 )'].astype(int)
# Find max and min density
max_density = density_2012.max()
min_density = density_2012.min()
# Calculate difference
difference = max_density - min_density
print(f"Final Answer: {difference}")