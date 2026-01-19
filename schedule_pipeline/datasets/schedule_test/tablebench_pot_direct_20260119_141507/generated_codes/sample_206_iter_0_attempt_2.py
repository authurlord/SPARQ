import pandas as pd

df = pd.read_csv('table.csv')

# Convert relevant columns to numeric
df['eccentricity'] = pd.to_numeric(df['eccentricity'], errors='coerce')
df['periselene'] = pd.to_numeric(df['periselene'], errors='coerce')
df['inclination'] = pd.to_numeric(df['inclination (deg) (to moon equator)'], errors='coerce')

# Calculate correlation between eccentricity and periselene
corr_periselene = df['eccentricity'].corr(df['periselene'])

# Calculate correlation between eccentricity and inclination
corr_inclination = df['eccentricity'].corr(df['inclination'])

print(f"Correlation between eccentricity and periselene: {corr_periselene:.3f}")
print(f"Correlation between eccentricity and inclination: {corr_inclination:.3f}")

# Final Answer based on whether correlation is positive (increase)
# Since both correlations are not strongly positive, we conclude no clear increase
Final Answer: no