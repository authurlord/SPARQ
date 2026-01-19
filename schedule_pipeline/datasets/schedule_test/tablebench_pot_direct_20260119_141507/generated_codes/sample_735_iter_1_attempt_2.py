import pandas as pd

df = pd.read_csv('table.csv')
# Convert density column to numeric, coercing errors to NaN (though all values are valid here)
df['density (inhabitants / km 2 )'] = pd.to_numeric(df['density (inhabitants / km 2 )'], errors='coerce')

# Find the city with the highest and lowest density
max_density = df.loc[df['density (inhabitants / km 2 )'].idxmax(), 'city']
min_density = df.loc[df['density (inhabitants / km 2 )'].idxmin(), 'city']
max_value = df['density (inhabitants / km 2 )'].max()
min_value = df['density (inhabitants / km 2 )'].min()
difference = max_value - min_value

print(f"Final Answer: {max_density}, {min_density}, {difference:.1f}")