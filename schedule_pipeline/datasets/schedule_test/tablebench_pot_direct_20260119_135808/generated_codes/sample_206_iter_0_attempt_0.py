import pandas as pd

df = pd.read_csv('table.csv')

# Convert relevant columns to float for numerical analysis
df['eccentricity'] = pd.to_numeric(df['eccentricity'])
df['periselene (km)'] = pd.to_numeric(df['periselene (km)'])
df['inclination (deg) (to moon equator)'] = pd.to_numeric(df['inclination (deg) (to moon equator)'])

# Calculate correlation between eccentricity and periselene
corr_ecc_peri = df['eccentricity'].corr(df['periselene (km)'])

# Calculate correlation between eccentricity and inclination
corr_ecc_incl = df['eccentricity'].corr(df['inclination (deg) (to moon equator)'])

print(f"Final Answer: {corr_ecc_peri:.3f}, {corr_ecc_incl:.3f}")