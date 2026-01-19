import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'pop density (per km2)' to numeric and find the max
max_density = df['pop density (per km2)'].max()
max_district = df[df['pop density (per km2)'] == max_density]['district'].values[0]

# Calculate average population density
avg_density = df['pop density (per km2)'].mean()

# Compute percentage difference
percentage_diff = ((max_density - avg_density) / avg_density) * 100

print(f"Final Answer: {max_district}, {percentage_diff:.2f}")