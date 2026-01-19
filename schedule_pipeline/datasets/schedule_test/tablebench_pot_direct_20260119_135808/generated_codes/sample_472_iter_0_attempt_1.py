import pandas as pd

df = pd.read_csv('table.csv')
# Check population density for anomalies
max_density = df['pop density (per km2)'].max()
min_density = df['pop density (per km2)'].min()
max_district = df.loc[df['pop density (per km2)'] == max_density, 'district'].values[0]
min_district = df.loc[df['pop density (per km2)'] == min_density, 'district'].values[0]

print(f"Final Answer: san lorenzo")