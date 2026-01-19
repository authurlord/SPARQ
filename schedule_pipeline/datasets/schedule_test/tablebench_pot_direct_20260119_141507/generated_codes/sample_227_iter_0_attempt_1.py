import pandas as pd

df = pd.read_csv('table.csv')
# Extract the relevant columns
density_mil = df['Population density (/mi²)']
land_area_mil = df['Land area (mi²)']

# Calculate the correlation
correlation = density_mil.corr(land_area_mil)
print(f"Final Answer: {correlation:.3f}")