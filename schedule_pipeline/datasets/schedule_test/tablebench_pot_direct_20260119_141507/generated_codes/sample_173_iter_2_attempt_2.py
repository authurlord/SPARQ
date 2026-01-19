import pandas as pd

df = pd.read_csv('table.csv')

# Extract relevant columns
eccentricity = df['eccentricity'].astype(float)
periselene = df['periselene (km)'].astype(float)
aposelene = df['aposelene (km)'].astype(float)

# Compute correlation
corr_periselene = df['eccentricity'].corr(df['periselene (km)'])
corr_aposelene = df['eccentricity'].corr(df['aposelene (km)'])

print(f"Correlation between eccentricity and periselene: {corr_periselene:.3f}")
print(f"Correlation between eccentricity and aposelene: {corr_aposelene:.3f}")

Final Answer: negative, positive