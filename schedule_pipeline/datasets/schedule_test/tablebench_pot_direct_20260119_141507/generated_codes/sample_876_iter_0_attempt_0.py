import pandas as pd

df = pd.read_csv('table.csv')
# Extract population density in 2012
density_2012 = df['population density 2012 (km 2 )']
max_density = density_2012.max()
min_density = density_2012.min()
difference = max_density - min_density
print(f"Final Answer: {difference}")