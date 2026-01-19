import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Extract the required columns
length_km = df['Length [km]']
basin_area_km2 = df['Drainage basin area [km2]']

# Compute the correlation coefficient (Pearson)
correlation = length_km.corr(basin_area_km2)

print(f"Final Answer: {correlation:.3f}")