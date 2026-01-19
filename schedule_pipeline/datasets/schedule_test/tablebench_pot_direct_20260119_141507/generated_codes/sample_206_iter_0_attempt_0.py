import pandas as pd

df = pd.read_csv('table.csv')

# Calculate correlation between eccentricity and periselene
corr_periselene = df['eccentricity'].corr(df['periselene'])

# Calculate correlation between eccentricity and inclination
corr_inclination = df['eccentricity'].corr(df['inclination'])

print(f"Correlation between eccentricity and periselene: {corr_periselene:.3f}")
print(f"Correlation between eccentricity and inclination: {corr_inclination:.3f}")
Final Answer: no