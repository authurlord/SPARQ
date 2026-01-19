import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'total' and 'hydroelectricity' to numeric, coercing errors to NaN
df['total'] = pd.to_numeric(df['total'], errors='coerce')
df['hydroelectricity'] = pd.to_numeric(df['hydroelectricity'], errors='coerce')

# Drop rows with NaN due to invalid parsing
df_clean = df.dropna(subset=['total', 'hydroelectricity'])

# Calculate correlation coefficient
correlation = df_clean['total'].corr(df_clean['hydroelectricity'])

print(f"Final Answer: {correlation:.3f}")