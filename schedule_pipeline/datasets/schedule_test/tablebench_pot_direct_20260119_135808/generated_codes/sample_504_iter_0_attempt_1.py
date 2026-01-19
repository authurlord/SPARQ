import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to appropriate data types
df['no of barangays'] = pd.to_numeric(df['no of barangays'])
df['area (km square)'] = pd.to_numeric(df['area (km square)'])

# Calculate barangay density
df['barangay_density'] = df['no of barangays'] / df['area (km square)']

# Find the city/municipality with the highest barangay density
max_density_city = df.loc[df['barangay_density'].idxmax(), 'city / municipality']

print(f"Final Answer: {max_density_city}")