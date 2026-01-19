import pandas as pd

df = pd.read_csv('table.csv')
# Extract population density column
pop_density = df['pop density (per km2)'].astype(float)

# Find the district with the highest population density
max_density = pop_density.max()
max_district = df.loc[pop_density.idxmax(), 'district']

# Calculate average population density
avg_density = pop_density.mean()

# Compute percentage difference
percentage_diff = ((max_density - avg_density) / avg_density) * 100

print(f"Final Answer: {max_district}, {percentage_diff:.2f}")