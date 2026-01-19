import pandas as pd

df = pd.read_csv('table.csv')
# Clean and convert 'typhus' and 'smallpox' columns to integers
df['typhus'] = df['typhus'].astype(str).str.replace(' ', '').astype(int)
df['smallpox'] = df['smallpox'].astype(str).str.replace(' ', '').str.replace('(avg)', '').astype(int)

# Calculate absolute difference between typhus and smallpox
df['diff'] = abs(df['typhus'] - df['smallpox'])

# Find the year with the minimum difference
min_diff_year = df.loc[df['diff'].idxmin(), 'year']
print(f"Final Answer: {min_diff_year}")