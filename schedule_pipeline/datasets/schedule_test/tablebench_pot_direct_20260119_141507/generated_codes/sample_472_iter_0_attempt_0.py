import pandas as pd

df = pd.read_csv('table.csv')
# Check for unusually high population density
density_values = df['pop density (per km2)']
# Find the maximum density and check if it's extremely high
max_density = density_values.max()
# Look at the district with max density
outlier_district = df.loc[density_values.idxmax(), 'district']
print(f"Final Answer: {outlier_district}")