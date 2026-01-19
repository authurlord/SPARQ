import pandas as pd

df = pd.read_csv('table.csv')
# Convert density column to numeric, coercing errors to NaN (though data seems clean)
df['density (inhabitants / km 2 )'] = pd.to_numeric(df['density (inhabitants / km 2 )'], errors='coerce')

# Find max and min density values and their corresponding cities
max_density = df.loc[df['density (inhabitants / km 2 )'].idxmax(), 'city']
min_density = df.loc[df['density (inhabitants / km 2 )'].idxmin(), 'city']
max_val = df['density (inhabitants / km 2 )'].max()
min_val = df['density (inhabitants / km 2 )'].min()
difference = max_val - min_val

print(f"Final Answer: {max_density}, {min_density}, {difference:.1f}")