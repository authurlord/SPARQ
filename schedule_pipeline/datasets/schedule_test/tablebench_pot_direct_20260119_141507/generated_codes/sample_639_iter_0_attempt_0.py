import pandas as pd

df = pd.read_csv('table.csv')
# Extract the columns of interest
area_km2 = df['area (km 2 )']
population = df['population']

# Calculate the correlation coefficient
correlation_coefficient = area_km2.corr(population)
print(f"Final Answer: {correlation_coefficient:.3f}")