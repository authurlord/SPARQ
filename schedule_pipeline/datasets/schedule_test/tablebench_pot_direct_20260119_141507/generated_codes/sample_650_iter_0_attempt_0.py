import pandas as pd

df = pd.read_csv('table.csv')
# Extract the relevant columns
area_sqkm = df['area in sqkm']
pop_density = df['population density 2012 (km 2)']

# Calculate the correlation coefficient
correlation_coefficient = area_sqkm.corr(pop_density)
print(f"Final Answer: {correlation_coefficient:.3f}")