import pandas as pd

df = pd.read_csv('table.csv')

# Clean 'total passengers' column: remove any non-numeric characters and convert to int
df['total passengers'] = df['total passengers'].str.replace(',', '').astype(int)

# Clean 'capacity in use' column: remove '%' and any commas, convert to float
df['capacity in use'] = df['capacity in use'].str.replace('%', '').str.replace(',', '').astype(float)

# Calculate correlation coefficient
correlation = df['total passengers'].corr(df['capacity in use'])

print(f"Final Answer: {correlation:.4f}")