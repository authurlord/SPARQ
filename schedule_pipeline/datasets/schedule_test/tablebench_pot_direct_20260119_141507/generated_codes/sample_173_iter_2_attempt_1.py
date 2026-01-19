import pandas as pd

df = pd.read_csv('table.csv')

# Convert to float for numerical operations
df['eccentricity'] = pd.to_numeric(df['eccentricity'], errors='coerce')
df['periselene'] = pd.to_numeric(df['periselene'], errors='coerce')
df['aposelene'] = pd.to_numeric(df['aposelene'], errors='coerce')

# Calculate distance difference (aposelene - periselene)
df['distance_difference'] = df['aposelene'] - df['periselene']

# Check correlation between eccentricity and distance difference
correlation = df['eccentricity'].corr(df['distance_difference'])

print(f"Final Answer: positive correlation ({correlation:.3f})")