import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Clean the 'smallpox' column by removing "(avg)" if present and convert to int
df['smallpox'] = df['smallpox'].astype(str).str.replace(r'\s*\(avg\)', '', regex=True).str.strip()
df['smallpox'] = pd.to_numeric(df['smallpox'], errors='coerce')

# Similarly, ensure typhus is numeric
df['typhus'] = pd.to_numeric(df['typhus'], errors='coerce')

# Drop rows where either typhus or smallpox is NaN (invalid)
df_clean = df.dropna(subset=['typhus', 'smallpox'])

# Compute absolute difference between typhus and smallpox
df_clean['difference'] = abs(df_clean['typhus'] - df_clean['smallpox'])

# Find the year with the minimum difference
min_diff_row = df_clean.loc[df_clean['difference'].idxmin()]
final_year = min_diff_row['year']

print(f"Final Answer: {final_year}")