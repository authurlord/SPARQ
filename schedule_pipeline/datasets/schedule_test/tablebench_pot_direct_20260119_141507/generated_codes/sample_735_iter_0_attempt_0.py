import pandas as pd

df = pd.read_csv('table.csv')
# Find the city with highest and lowest density
max_density_row = df.loc[df['density (inhabitants / km 2 )'].idxmax()]
min_density_row = df.loc[df['density (inhabitants / km 2 )'].idxmin()]

highest_density_city = max_density_row['city']
lowest_density_city = min_density_row['city']
density_difference = max_density_row['density (inhabitants / km 2 )'] - min_density_row['density (inhabitants / km 2 )']

print(f"Final Answer: {highest_density_city}, {lowest_density_city}, {density_difference:.1f}")