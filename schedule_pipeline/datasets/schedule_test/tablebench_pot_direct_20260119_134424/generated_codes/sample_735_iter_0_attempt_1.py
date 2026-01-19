import pandas as pd

df = pd.read_csv('table.csv')
# Convert density column to numeric for comparison
df['density (inhabitants / km 2 )'] = pd.to_numeric(df['density (inhabitants / km 2 )'])
# Find city with highest density
highest_density_city = df.loc[df['density (inhabitants / km 2 )'].idxmax(), 'city']
# Find city with lowest density
lowest_density_city = df.loc[df['density (inhabitants / km 2 )'].idxmin(), 'city']
# Calculate difference
density_difference = df['density (inhabitants / km 2 )'].max() - df['density (inhabitants / km 2 )'].min()
print(f"Final Answer: {highest_density_city}, {density_difference:.1f}")