import pandas as pd

df = pd.read_csv('table.csv')

# Clean 'typhus' and 'smallpox' columns: remove non-numeric characters and convert to int
df['typhus'] = df['typhus'].astype(str).str.replace(r'[^0-9]', '', regex=True).astype(int)
df['smallpox'] = df['smallpox'].astype(str).str.replace(r'[^0-9]', '', regex=True).astype(int)

# Calculate absolute difference between typhus and smallpox
df['diff'] = (df['typhus'] - df['smallpox']).abs()

# Find the year with the minimum difference
min_diff_year = df.loc[df['diff'].idxmin(), 'year']

print(f"Final Answer: {min_diff_year}")