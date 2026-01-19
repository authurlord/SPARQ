import pandas as pd

df = pd.read_csv('table.csv')
# Correctly extract the columns with proper spacing
area_sqkm = df['area in sqkm']
pop_density = df['population density 2012 (km 2)']

# Calculate the correlation coefficient
correlation = area_sqkm.corr(pop_density)
print(f"Final Answer: {correlation:.3f}")