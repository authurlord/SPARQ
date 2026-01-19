import pandas as pd

df = pd.read_csv('table.csv')
# Clean and convert 'total passengers' and 'capacity' columns to numeric
df['total passengers'] = df['total passengers'].str.replace(',', '').astype(float)
df['capacity'] = df['capacity'].str.replace(',', '').astype(float)

# Calculate correlation coefficient
correlation = df['total passengers'].corr(df['capacity'])
print(f"Final Answer: {correlation:.4f}")