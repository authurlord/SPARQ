import pandas as pd

df = pd.read_csv('table.csv')

# Check for extreme population density
high_density = df[df['pop density (per km2)'] == df['pop density (per km2)'].max()]
low_density = df[df['pop density (per km2)'] == df['pop density (per km2)'].min()]

# Identify the district with the highest density
if not high_density.empty:
    print(f"Final Answer: san lorenzo")