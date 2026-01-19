import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'pop density (per km2)' to numeric
df['pop density (per km2)'] = pd.to_numeric(df['pop density (per km2)'])

# Find the district with the highest population density
max_density_district = df.loc[df['pop density (per km2)'].idxmax()]
max_density = max_density_district['pop density (per km2)']
district_name = max_density_district['district']

# Calculate average population density
avg_density = df['pop density (per km2)'].mean()

# Calculate percentage difference
percentage_diff = ((max_density - avg_density) / avg_density) * 100

print(f"Final Answer: {district_name}, {percentage_diff:.1f}")