import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Clean and convert relevant columns to numeric
arrivals_col = 'internl tourist arrivals 2011 (x1000)'
receipts_col = 'internl tourism receipts 2011 (million usd )'

# Replace 'n / d' with NaN and convert to numeric
df[arrivals_col] = pd.to_numeric(df[arrivals_col].replace('n / d', np.nan), errors='coerce')
df[receipts_col] = pd.to_numeric(df[receipts_col].replace('n / d', np.nan), errors='coerce')

# Drop rows with missing values in either column
df_clean = df.dropna(subset=[arrivals_col, receipts_col])

# Compute correlation
correlation = df_clean[arrivals_col].corr(df_clean[receipts_col])

print(f"Final Answer: {correlation:.3f}")