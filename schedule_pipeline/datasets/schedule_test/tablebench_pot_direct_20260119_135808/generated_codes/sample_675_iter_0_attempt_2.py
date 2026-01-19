import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'total passengers' to numeric, removing any non-numeric characters
df['total passengers'] = pd.to_numeric(df['total passengers'], errors='coerce')

# Convert 'capacity in use' to numeric, removing '%' and handling any commas
df['capacity in use'] = df['capacity in use'].str.replace('%', '').str.replace(',', '').astype(float) / 100

# Calculate correlation coefficient
correlation = df['total passengers'].corr(df['capacity in use'])

print(f"Final Answer: {correlation:.4f}")