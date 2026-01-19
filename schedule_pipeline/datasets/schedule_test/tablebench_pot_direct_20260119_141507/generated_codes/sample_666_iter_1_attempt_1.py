import pandas as pd

df = pd.read_csv('table.csv')
# Correct column names based on the table
area_ha = df['area (ha)']
pop_density = df['pop density (per km2)']

# Calculate the correlation coefficient
correlation_coefficient = area_ha.corr(pop_density)
print(f"Final Answer: {correlation_coefficient:.3f}")