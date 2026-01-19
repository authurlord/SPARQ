import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total' and 'hydroelectricity' columns to numeric, coercing errors to NaN
df['total'] = pd.to_numeric(df['total'], errors='coerce')
df['hydroelectricity'] = pd.to_numeric(df['hydroelectricity'], errors='coerce')

# Drop rows with missing values in either column
df.dropna(subset=['total', 'hydroelectricity'], inplace=True)

# Calculate correlation coefficient
correlation = df['total'].corr(df['hydroelectricity'])
print(f"Final Answer: {correlation:.3f}")