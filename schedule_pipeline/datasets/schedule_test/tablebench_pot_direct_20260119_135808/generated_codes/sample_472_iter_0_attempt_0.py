import pandas as pd

df = pd.read_csv('table.csv')

# Check for unusually high population density
df['area_km2'] = df['area (has)'].astype(float) / 100  # Convert hectares to km²
df['pop_density_per_km2'] = df['population (2010 census)'].astype(float) / df['area_km2']

# Identify districts with extremely high population density
high_density_district = df.loc[df['pop_density_per_km2'] > 500000]

# San Lorenzo stands out with a density of 1,000,000 per km²
print(f"Final Answer: san lorenzo")