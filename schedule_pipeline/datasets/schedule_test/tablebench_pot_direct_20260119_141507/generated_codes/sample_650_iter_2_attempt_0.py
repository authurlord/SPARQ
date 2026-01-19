import pandas as pd

df = pd.read_csv('table.csv')
# Extract the required columns
area_sqkm = df['area in sqkm']
pop_density = df['population density 2012 (km 2)']

# Calculate the correlation coefficient
correlation = area_sqkm.corr(pop_density)
print(f"Final Answer: {correlation:.3f}")