import pandas as pd

df = pd.read_csv('table.csv')
# Extract the required columns
area_km2 = df['area (km 2 )'].astype(float)
population = df['population'].astype(int)

# Calculate the correlation coefficient
correlation_coefficient = area_km2.corr(population)
print(f"Final Answer: {correlation_coefficient:.3f}")