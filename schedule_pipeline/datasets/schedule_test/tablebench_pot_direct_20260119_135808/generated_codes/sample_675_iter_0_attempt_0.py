import pandas as pd

df = pd.read_csv('table.csv')
# Clean and convert 'total passengers' to numeric
df['total passengers'] = pd.to_numeric(df['total passengers'], errors='coerce')

# Clean and convert 'capacity in use' to numeric, removing '%' and handling commas
df['capacity in use'] = df['capacity in use'].str.replace('%', '').str.replace(',', '').astype(float) / 100

# Calculate correlation coefficient
correlation = df['total passengers'].corr(df['capacity in use'])

print(f"Final Answer: {correlation:.4f}")