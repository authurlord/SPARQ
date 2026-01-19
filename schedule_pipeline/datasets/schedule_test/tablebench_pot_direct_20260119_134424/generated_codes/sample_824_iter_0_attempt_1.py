import pandas as pd

df = pd.read_csv('table.csv')
# Convert population density to float
df['pop density (per km2)'] = pd.to_numeric(df['pop density (per km2)'])

# Find the district with the highest population density
max_density_row = df.loc[df['pop density (per km2)'].idxmax()]
highest_district = max_density_row['district']
max_density = max_density_row['pop density (per km2)']

# Calculate average population density
avg_density = df['pop density (per km2)'].mean()

# Calculate percentage difference
percentage_diff = ((max_density - avg_density) / avg_density) * 100

print(f"Final Answer: {highest_district}, {percentage_diff:.1f}")