import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Clean data: replace 'n / d' with NaN
df['internl tourism receipts 2011 (million usd )'] = df['internl tourism receipts 2011 (million usd )'].replace('n / d', np.nan)
df['internl tourist arrivals 2011 (x1000)'] = pd.to_numeric(df['internl tourist arrivals 2011 (x1000)'], errors='coerce')

# Drop rows with missing values in either column
df_clean = df.dropna(subset=['internl tourist arrivals 2011 (x1000)', 'internl tourism receipts 2011 (million usd )'])

# Calculate correlation between tourist arrivals and receipts
correlation = df_clean['internl tourist arrivals 2011 (x1000)'].corr(df_clean['internl tourism receipts 2011 (million usd )'])

print(f"Final Answer: {correlation:.3f}")