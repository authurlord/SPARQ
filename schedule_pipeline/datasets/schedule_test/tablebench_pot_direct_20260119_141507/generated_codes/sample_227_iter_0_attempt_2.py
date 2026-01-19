import pandas as pd

df = pd.read_csv('table.csv')
# Extract the relevant columns
density_mi2 = df['Population density (/mi²)']
land_area_mi2 = df['Land area (mi²)']

# Compute the correlation
correlation = density_mi2.corr(land_area_mi2)
print(f"Final Answer: {correlation:.3f}")