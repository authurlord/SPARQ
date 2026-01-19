import pandas as pd

df = pd.read_csv('table.csv')
# Extract the relevant columns
area_sqkm = df['area in sqkm']
density_2012 = df['population density 2012 (km 2)']

# Calculate the correlation coefficient
correlation = area_sqkm.corr(density_2012)
print(f"Final Answer: {correlation:.3f}")