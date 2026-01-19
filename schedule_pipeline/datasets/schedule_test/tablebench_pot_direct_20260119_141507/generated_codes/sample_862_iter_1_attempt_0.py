import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert relevant columns to numeric, handling strings with parentheses
def extract_numeric(x):
    if isinstance(x, str):
        # Extract number before any text (e.g., "2940 (avg)" -> 2940)
        import re
        match = re.search(r'^(\d+)', x)
        return int(match.group(1)) if match else np.nan
    return x

# Apply extraction to typhus and smallpox columns
df['typhus'] = df['typhus'].apply(extract_numeric)
df['smallpox'] = df['smallpox'].apply(extract_numeric)

# Drop rows where either typhus or smallpox is NaN
df = df.dropna(subset=['typhus', 'smallpox'])

# Calculate absolute difference between typhus and smallpox
df['difference'] = abs(df['typhus'] - df['smallpox'])

# Find the year with the minimum difference
min_diff_row = df.loc[df['difference'].idxmin()]
final_year = min_diff_row['year']

print(f"Final Answer: {final_year}")