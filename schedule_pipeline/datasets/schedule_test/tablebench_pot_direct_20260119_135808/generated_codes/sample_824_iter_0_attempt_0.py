import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'pop density (per km2)' to numeric
df['pop density (per km2)'] = pd.to_numeric(df['pop density (per km2)'])

# Find the district with the highest population density
max_density_district = df.loc[df['pop density (per km2)'].idxmax()]

# Calculate average population density
average_density = df['pop density (per km2)'].mean()

# Calculate percentage difference
percentage_difference = ((max_density_district['pop density (per km2)'] - average_density) / average_density) * 100

# Output the results
print(f"Final Answer: {max_density_district['district']}, {percentage_difference:.1f}")