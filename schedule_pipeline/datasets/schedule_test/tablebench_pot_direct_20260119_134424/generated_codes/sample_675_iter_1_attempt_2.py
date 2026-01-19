import pandas as pd

df = pd.read_csv('table.csv')

# Clean 'capacity in use' column: remove spaces and convert to float
df['capacity in use'] = df['capacity in use'].str.replace(' ', '').astype(float)

# Convert 'total passengers' to float
df['total passengers'] = df['total passengers'].astype(float)

# Calculate correlation coefficient
correlation = df['total passengers'].corr(df['capacity in use'])

print(f"Final Answer: {correlation:.4f}")