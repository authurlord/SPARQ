import pandas as pd

df = pd.read_csv('table.csv')
# Convert rank to numeric and reverse it (lower rank is better)
df['rank_numeric'] = pd.to_numeric(df['rank'], errors='coerce')
df['rank_reversed'] = 18 - df['rank_numeric']  # assuming max rank is 18 for this dataset

# Extract international passengers as numeric
df['international passengers'] = pd.to_numeric(df['international passengers'].str.replace('%', ''), errors='coerce')

# Calculate correlation between reversed rank and international passengers
correlation = df['international passengers'].corr(df['rank_reversed'])

print(f"Final Answer: {correlation:.2f}")