import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'total passengers' and 'capacity' to numeric (already numeric in format)
df['total passengers'] = pd.to_numeric(df['total passengers'].str.replace(',', ''), errors='coerce')
df['capacity'] = pd.to_numeric(df['capacity'], errors='coerce')

# Drop rows with NaN due to conversion issues
df_clean = df.dropna(subset=['total passengers', 'capacity'])

# Calculate correlation coefficient
correlation = df_clean['total passengers'].corr(df_clean['capacity'])

print(f"Final Answer: {correlation:.2f}")