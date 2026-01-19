import pandas as pd

df = pd.read_csv('table.csv')
# Extract population density in 2012
density_2012 = df['population density 2012 (km 2 )'].astype(int)
# Calculate difference between highest and lowest
density_difference = density_2012.max() - density_2012.min()
print(f"Final Answer: {density_difference}")