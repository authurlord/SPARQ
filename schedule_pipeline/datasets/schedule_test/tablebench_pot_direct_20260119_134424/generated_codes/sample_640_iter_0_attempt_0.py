import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'size (cents)' and 'error' columns to numeric
df['size (cents)'] = pd.to_numeric(df['size (cents)'], errors='coerce')
df['error'] = pd.to_numeric(df['error'], errors='coerce')

# Calculate correlation coefficient
correlation = df['size (cents)'].corr(df['error'])

print(f"Final Answer: {correlation:.4f}")