import pandas as pd

df = pd.read_csv('table.csv')

# Clean 'total passengers' column: remove commas and convert to float
df['total passengers'] = df['total passengers'].str.replace(',', '').astype(float)

# Clean 'capacity in use' column: remove '%' and convert to float
df['capacity in use'] = df['capacity in use'].str.replace('%', '').str.replace(',', '').astype(float) / 100

# Calculate correlation coefficient
correlation = df['total passengers'].corr(df['capacity in use'])

print(f"Final Answer: {correlation:.4f}")