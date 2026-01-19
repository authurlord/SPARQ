import pandas as pd

df = pd.read_csv('table.csv')
# Correctly access columns with spaces and special characters using quotes
area_km2 = df['area in sqkm']
density_km2 = df['population density 2012 (km 2)']

# Calculate correlation coefficient
correlation = area_km2.corr(density_km2)
print(f"Final Answer: {correlation:.3f}")