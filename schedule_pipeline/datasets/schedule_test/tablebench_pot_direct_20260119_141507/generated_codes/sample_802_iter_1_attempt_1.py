import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'rank' to numeric, coercing errors to NaN if any
df['rank'] = pd.to_numeric(df['rank'], errors='coerce')

# Filter rows where rank >= 10
filtered_df = df[df['rank'] >= 10]

# Extract televotes for filtered rows
televotes_filtered = filtered_df['televotes'].dropna()

# Compute median
median_televotes = televotes_filtered.median()
print(f"Final Answer: {median_televotes}")