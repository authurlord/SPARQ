import pandas as pd

df = pd.read_csv('table.csv')
# Convert relevant columns to numeric
df['no of barangays'] = pd.to_numeric(df['no of barangays'], errors='coerce')
df['area (km square)'] = pd.to_numeric(df['area (km square)'], errors='coerce')

# Calculate barangay density
df['barangay_density'] = df['no of barangays'] / df['area (km square)']

# Find the city with the highest density
max_density_city = df.loc[df['barangay_density'].idxmax(), 'city / municipality']
print(f"Final Answer: {max_density_city}")