import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract relevant columns
eccentricity = df['eccentricity'].astype(float)
periselene = df['periselene (km)'].astype(float)
inclination = df['inclination (deg) (to moon equator)'].astype(float)

# Calculate correlation coefficients
corr_periselene = eccentricity.corr(periselene)
corr_inclination = eccentricity.corr(inclination)

print(f"Correlation between eccentricity and periselene: {corr_periselene:.3f}")
print(f"Correlation between eccentricity and inclination: {corr_inclination:.3f}")

# Determine if there's a positive correlation (greater than 0.3, say)
if corr_periselene > 0.3:
    print("Final Answer: periselene")
elif corr_inclination > 0.3:
    print("Final Answer: inclination")
else:
    print("Final Answer: neither")