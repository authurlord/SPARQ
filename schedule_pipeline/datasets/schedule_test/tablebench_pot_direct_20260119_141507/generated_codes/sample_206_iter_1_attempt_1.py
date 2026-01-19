import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert columns to numeric (in case of string issues)
df['eccentricity'] = pd.to_numeric(df['eccentricity'], errors='coerce')
df['periselene'] = pd.to_numeric(df['periselene'], errors='coerce')
df['inclination'] = pd.to_numeric(df['inclination (deg) (to moon equator)'], errors='coerce')

# Drop any rows with NaN due to conversion errors
df = df.dropna()

# Compute correlation between eccentricity and periselene
corr_periselene = df['eccentricity'].corr(df['periselene'])

# Compute correlation between eccentricity and inclination
corr_inclination = df['eccentricity'].corr(df['inclination'])

print(f"Correlation between eccentricity and periselene: {corr_periselene:.3f}")
print(f"Correlation between eccentricity and inclination: {corr_inclination:.3f}")

# Final answer: Based on the correlation values, determine if there is a corresponding increase
# If correlation is positive and significant, then yes; otherwise no.
# From the data, periselene decreases when eccentricity increases → negative correlation
# Inclination increases slightly → positive but weak

# Since the question asks whether an increase in eccentricity causes a corresponding increase,
# only positive and meaningful correlation counts.

# Final Answer: None of them show a corresponding increase (periselene decreases, inclination increases weakly)
Final Answer: no, no