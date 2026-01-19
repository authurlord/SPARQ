import pandas as pd

df = pd.read_csv('table.csv')

# Calculate the absolute difference between typhus and smallpox cases
df['difference'] = abs(df['typhus'] - df['smallpox'])

# Find the year with the minimum difference
min_diff_year = df.loc[df['difference'].idxmin(), 'year']

print(f"Final Answer: {min_diff_year}")