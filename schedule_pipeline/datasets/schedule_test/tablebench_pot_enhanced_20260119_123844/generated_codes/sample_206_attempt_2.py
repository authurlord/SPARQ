import pandas as pd

df = pd.read_csv('table.csv')
# Convert relevant columns to numeric
df['eccentricity'] = pd.to_numeric(df['eccentricity'])
df['periselene (km)'] = pd.to_numeric(df['periselene (km)'])
df['inclination (deg) (to moon equator)'] = pd.to_numeric(df['inclination (deg) (to moon equator)'])

# Calculate correlation coefficients
corr_periselene = df['eccentricity'].corr(df['periselene (km)'])
corr_inclination = df['eccentricity'].corr(df['inclination (deg) (to moon equator)'])

# Output results
print(f"Correlation between eccentricity and periselene: {corr_periselene:.3f}")
print(f"Correlation between eccentricity and inclination: {corr_inclination:.3f}")

# Final answer based on whether either correlation is positive and significant
if corr_periselene > 0.5:
    final_answer = "yes"
elif corr_inclination > 0.5:
    final_answer = "yes"
else:
    final_answer = "no"

print(f"Final Answer: {final_answer}")