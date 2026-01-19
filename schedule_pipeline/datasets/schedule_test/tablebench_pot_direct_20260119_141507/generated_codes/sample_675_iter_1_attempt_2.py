import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Clean the 'capacity in use' column: replace comma with dot and remove %
df['capacity in use'] = df['capacity in use'].str.replace(',', '.').str.replace('%', '')

# Convert to numeric (handles any remaining parsing issues)
df['total passengers'] = pd.to_numeric(df['total passengers'], errors='coerce')
df['capacity in use'] = pd.to_numeric(df['capacity in use'], errors='coerce')

# Drop rows with NaN values due to invalid entries
df = df.dropna()

# Calculate correlation between 'total passengers' and 'capacity in use'
correlation = df['total passengers'].corr(df['capacity in use'])

print(f"Final Answer: {correlation:.2f}")