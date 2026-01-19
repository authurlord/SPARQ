import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'typhus' and 'smallpox' to numeric, coercing errors to NaN
df['typhus'] = pd.to_numeric(df['typhus'], errors='coerce')
df['smallpox'] = pd.to_numeric(df['smallpox'], errors='coerce')

# Drop rows with NaN values (due to invalid entries)
df_clean = df.dropna(subset=['typhus', 'smallpox'])

# Compute the correlation coefficient
correlation = df_clean['typhus'].corr(df_clean['smallpox'])

print(f"Final Answer: {correlation:.3f}")