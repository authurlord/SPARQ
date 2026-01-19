import pandas as pd

df = pd.read_csv('table.csv')

# Extract relevant columns
eccentricity = df['eccentricity'].astype(float)
periselene = df['periselene (km)'].astype(float)
aposelene = df['aposelene (km)'].astype(float)

# Compute correlations
correlation_periselene = periselene.corr(eccentricity)
correlation_apselene = aposelene.corr(eccentricity)

print(f"Correlation between eccentricity and periselene: {correlation_periselene:.3f}")
print(f"Correlation between eccentricity and aposelene: {correlation_apselene:.3f}")

Final Answer: negative, positive