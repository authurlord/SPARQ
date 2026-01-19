import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert relevant columns to numeric (in case of string issues)
df['eccentricity'] = pd.to_numeric(df['eccentricity'], errors='coerce')
df['periselene'] = pd.to_numeric(df['periselene'], errors='coerce')
df['inclination (deg) (to moon equator)'] = pd.to_numeric(df['inclination (deg) (to moon equator)'], errors='coerce')

# Drop rows with NaN due to conversion errors
df = df.dropna()

# Compute correlation between eccentricity and periselene
corr_periselene = df['eccentricity'].corr(df['periselene'])

# Compute correlation between eccentricity and inclination
corr_inclination = df['eccentricity'].corr(df['inclination (deg) (to moon equator)'])

# Determine which has a positive correlation (greater than 0)
if corr_periselene > 0 and corr_inclination > 0:
    # Both are positively correlated; pick the one with higher correlation
    if abs(corr_periselene) > abs(corr_inclination):
        result = "periselene"
    else:
        result = "inclination (deg) (to moon equator)"
elif corr_periselene > 0:
    result = "periselene"
elif corr_inclination > 0:
    result = "inclination (deg) (to moon equator)"
else:
    result = "neither"

print(f"Final Answer: {result}")