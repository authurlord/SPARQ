import pandas as pd

df = pd.read_csv('table.csv')
# Convert density column to numeric, coercing errors to NaN (though all values should be valid)
df['density (inhabitants / km 2 )'] = pd.to_numeric(df['density (inhabitants / km 2 )'], errors='coerce')

# Find the city with the highest and lowest density
max_density = df['density (inhabitants / km 2 )'].max()
min_density = df['density (inhabitants / km 2 )'].min()

city_highest = df[df['density (inhabitants / km 2 )'] == max_density]['city'].values[0]
city_lowest = df[df['density (inhabitants / km 2 )'] == min_density]['city'].values[0]

difference = max_density - min_density

print(f"Final Answer: {city_highest}, {difference:.1f}")