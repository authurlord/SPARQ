import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert columns to numeric, coercing errors to NaN
df['typhus'] = pd.to_numeric(df['typhus'], errors='coerce')
df['smallpox'] = pd.to_numeric(df['smallpox'], errors='coerce')

# Drop rows where either typhus or smallpox is NaN
df_filtered = df.dropna(subset=['typhus', 'smallpox'])

# Calculate the absolute difference between typhus and smallpox
df_filtered['diff'] = abs(df_filtered['typhus'] - df_filtered['smallpox'])

# Find the year with the minimum difference
min_diff_row = df_filtered.loc[df_filtered['diff'].idxmin()]
final_year = min_diff_row['year']

print(f"Final Answer: {final_year}")