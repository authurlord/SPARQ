import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'total' and 'hydroelectricity' to numeric, handling invalid entries
df['total'] = pd.to_numeric(df['total'], errors='coerce')
df['hydroelectricity'] = pd.to_numeric(df['hydroelectricity'].str.replace(r'\([^)]*\)', '', regex=True), errors='coerce')

# Drop rows with NaN after conversion
df_clean = df.dropna(subset=['total', 'hydroelectricity'])

# Compute correlation
correlation = df_clean['total'].corr(df_clean['hydroelectricity'])

print(f"Final Answer: {correlation:.3f}")