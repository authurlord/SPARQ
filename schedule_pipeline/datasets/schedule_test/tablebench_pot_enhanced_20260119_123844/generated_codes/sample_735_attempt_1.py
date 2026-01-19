import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'density (inhabitants / km 2 )' to float for comparison
df['density (inhabitants / km 2 )'] = pd.to_numeric(df['density (inhabitants / km 2 )'])
# Find the city with the highest density
max_density_city = df.loc[df['density (inhabitants / km 2 )'].idxmax(), 'city']
# Find the city with the lowest density
min_density_city = df.loc[df['density (inhabitants / km 2 )'].idxmin(), 'city']
# Calculate the difference
density_difference = df['density (inhabitants / km 2 )'].max() - df['density (inhabitants / km 2 )'].min()
print(f"Final Answer: {max_density_city}, {density_difference:.1f}")