import pandas as pd

df = pd.read_csv('table.csv')

# Clean the 'capacity in use' column by removing commas and spaces and converting to float
df['capacity in use'] = df['capacity in use'].str.replace(',', '').str.replace(' ', '').astype(float)

# Compute the correlation between 'total passengers' and 'capacity in use'
correlation = df['total passengers'].corr(df['capacity in use'])

print(f"Final Answer: {correlation:.2f}")