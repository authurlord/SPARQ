import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'total passengers' and 'capacity' to numeric
df['total passengers'] = pd.to_numeric(df['total passengers'].str.replace(',', ''), errors='coerce')
df['capacity'] = pd.to_numeric(df['capacity'].str.replace(',', ''), errors='coerce')

# Drop rows with NaN due to invalid parsing
df = df.dropna(subset=['total passengers', 'capacity'])

# Calculate correlation coefficient
correlation = df['total passengers'].corr(df['capacity'])
print(f"Final Answer: {correlation:.2f}")