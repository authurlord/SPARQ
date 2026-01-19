import pandas as pd

df = pd.read_csv('table.csv')
# Extract population density for 2012
density_2012 = df['population density 2012 (km 2 )']
# Calculate difference between highest and lowest density
difference = density_2012.max() - density_2012.min()
print(f"Final Answer: {difference}")