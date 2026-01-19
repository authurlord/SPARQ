import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric, coercing errors to NaN
df['area (km square)'] = pd.to_numeric(df['area (km square)'], errors='coerce')
df['pop'] = pd.to_numeric(df['pop'], errors='coerce')

# Calculate correlation coefficient
correlation = df['area (km square)'].corr(df['pop'])

print(f"Final Answer: {correlation:.4f}")