import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'no of barangays' and 'area (km square)' to numeric
df['no of barangays'] = pd.to_numeric(df['no of barangays'])
df['area (km square)'] = pd.to_numeric(df['area (km square)'])

# Calculate barangay density
df['barangay density'] = df['no of barangays'] / df['area (km square)']

# Find the city/municipality with the highest barangay density
max_density_city = df.loc[df['barangay density'].idxmax(), 'city / municipality']

print(f"Final Answer: {max_density_city}")