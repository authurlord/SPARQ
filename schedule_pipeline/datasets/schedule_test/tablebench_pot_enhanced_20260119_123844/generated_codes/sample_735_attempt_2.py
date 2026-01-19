import pandas as pd

df = pd.read_csv('table.csv')
# Convert density column to numeric
df['density (inhabitants / km 2 )'] = pd.to_numeric(df['density (inhabitants / km 2 )'])
# Find the city with highest and lowest density
max_density_city = df.loc[df['density (inhabitants / km 2 )'].idxmax(), 'city']
min_density_city = df.loc[df['density (inhabitants / km 2 )'].idxmin(), 'city']
# Calculate the difference
density_difference = df['density (inhabitants / km 2 )'].max() - df['density (inhabitants / km 2 )'].min()
print(f"Final Answer: {max_density_city}, {density_difference:.1f}")