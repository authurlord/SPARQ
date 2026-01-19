import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Clean data: remove rows where either arrival or receipts is 'n / d'
df_clean = df.dropna(subset=['internl tourist arrivals 2011 (x1000)', 'internl tourism receipts 2011 (million usd)'])

# Convert the arrival and receipts columns to numeric
df_clean['internl tourist arrivals 2011 (x1000)'] = pd.to_numeric(df_clean['internl tourist arrivals 2011 (x1000)'], errors='coerce')
df_clean['internl tourism receipts 2011 (million usd)'] = pd.to_numeric(df_clean['internl tourism receipts 2011 (million usd)'], errors='coerce')

# Drop any remaining NaN values
df_clean = df_clean.dropna()

# Compute correlation
correlation = df_clean['internl tourist arrivals 2011 (x1000)'].corr(df_clean['internl tourism receipts 2011 (million usd)'])

print(f"Final Answer: {correlation:.3f}")