import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['eccentricity'] = pd.to_numeric(df['eccentricity'])
df['period (h)'] = pd.to_numeric(df['period (h)'])

# Calculate correlation coefficient
correlation = df['eccentricity'].corr(df['period (h)'])
print(f"Final Answer: {correlation:.4f}")