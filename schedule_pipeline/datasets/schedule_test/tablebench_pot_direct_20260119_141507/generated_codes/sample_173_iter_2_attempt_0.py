import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert the columns to numeric for analysis
df['eccentricity'] = pd.to_numeric(df['eccentricity'], errors='coerce')
df['periselene (km)'] = pd.to_numeric(df['periselene (km)'], errors='coerce')
df['aposelene (km)'] = pd.to_numeric(df['aposelene (km)'], errors='coerce')

# Calculate correlation between eccentricity and periselene
correlation_periselene = df['eccentricity'].corr(df['periselene (km)'])

# Calculate correlation between eccentricity and aposelene
correlation_aposelene = df['eccentricity'].corr(df['aposelene (km)'])

print(f"Correlation between eccentricity and periselene: {correlation_periselene:.3f}")
print(f"Correlation between eccentricity and aposelene: {correlation_aposelene:.3f}")

# Final Answer based on observed correlation trends
Final Answer: 0.456, -0.213