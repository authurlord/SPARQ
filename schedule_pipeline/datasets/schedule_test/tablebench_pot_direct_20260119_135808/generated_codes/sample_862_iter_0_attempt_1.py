import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'typhus' and 'smallpox' to numeric, handling non-numeric values (e.g., '2940 (avg)')
df['typhus'] = pd.to_numeric(df['typhus'], errors='coerce')
df['smallpox'] = pd.to_numeric(df['smallpox'], errors='coerce')

# Calculate absolute difference
df['diff'] = (df['typhus'] - df['smallpox']).abs()

# Find the year with minimum difference
min_diff_year = df.loc[df['diff'].idxmin(), 'year']
print(f"Final Answer: {min_diff_year}")