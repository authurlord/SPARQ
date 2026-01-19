import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Clean data: convert to numeric and drop NaNs
df['internl tourist arrivals 2011 (x1000)'] = pd.to_numeric(df['internl tourist arrivals 2011 (x1000)'], errors='coerce')
df['internl tourism receipts 2011 (million usd )'] = pd.to_numeric(df['internl tourism receipts 2011 (million usd )'], errors='coerce')

# Drop rows with missing values in either column
df_clean = df.dropna(subset=['internl tourist arrivals 2011 (x1000)', 'internl tourism receipts 2011 (million usd )'])

# Calculate correlation
correlation = df_clean['internl tourism receipts 2011 (million usd )'].corr(df_clean['internl tourist arrivals 2011 (x1000)'])

print(f"Final Answer: {correlation:.3f}")