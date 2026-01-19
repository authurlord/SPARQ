import pandas as pd

df = pd.read_csv('table.csv')

# Convert columns to numeric
df['eccentricity'] = pd.to_numeric(df['eccentricity'], errors='coerce')
df['periselene'] = pd.to_numeric(df['periselene'], errors='coerce')
df['aposelene'] = pd.to_numeric(df['aposelene'], errors='coerce')

# Calculate correlation between eccentricity and periselene/aposelene
correlation_periselene = df['eccentricity'].corr(df['periselene'])
correlation_apselene = df['eccentricity'].corr(df['aposelene'])

print(f"Correlation between eccentricity and periselene: {correlation_periselene:.3f}")
print(f"Correlation between eccentricity and aposelene: {correlation_apselene:.3f}")

Final Answer: negative, positive