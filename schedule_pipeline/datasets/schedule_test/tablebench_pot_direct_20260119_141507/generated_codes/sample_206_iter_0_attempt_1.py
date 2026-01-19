import pandas as pd

df = pd.read_csv('table.csv')

# Extract relevant columns
eccentricity = df['eccentricity'].astype(float)
periselene = df['periselene (km)'].astype(float)
inclination = df['inclination (deg) (to moon equator)'].astype(float)

# Compute correlation coefficients
corr_periselene = eccentricity.corr(periselene)
corr_inclination = eccentricity.corr(inclination)

print(f"Correlation between eccentricity and periselene: {corr_periselene:.3f}")
print(f"Correlation between eccentricity and inclination: {corr_inclination:.3f}")

# Final Answer: Based on the correlation values, determine if there's a corresponding increase
# If both are positive, then yes; otherwise, no.
if corr_periselene > 0:
    periselene_increase = "yes"
else:
    periselene_increase = "no"

if corr_inclination > 0:
    inclination_increase = "yes"
else:
    inclination_increase = "no"

Final Answer: periselene_increase, inclination_increase