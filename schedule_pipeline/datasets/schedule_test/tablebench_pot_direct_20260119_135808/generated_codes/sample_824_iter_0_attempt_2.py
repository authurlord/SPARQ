import pandas as pd

df = pd.read_csv('table.csv')
# Convert population density to float
df['pop density (per km2)'] = pd.to_numeric(df['pop density (per km2)'])

# Find the district with the highest population density
max_density_district = df.loc[df['pop density (per km2)'].idxmax()]
highest_density = max_density_district['pop density (per km2)']
district_name = max_density_district['district']

# Calculate average population density
average_density = df['pop density (per km2)'].mean()

# Calculate percentage difference
percentage_difference = ((highest_density - average_density) / average_density) * 100

print(f"Final Answer: {district_name}, {percentage_difference:.2f}")