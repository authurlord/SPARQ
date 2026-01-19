import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Year' to integer for proper ordering
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Sort by Year to ensure chronological order
df = df.sort_values('Year').reset_index(drop=True)

# Extract Political Rights scores
political_rights = df['Political Rights'].astype(int)

# Compute differences between consecutive years
differences = political_rights[1:] - political_rights[:-1]

# Find the first index where the difference is <= -2 (decrease of at least 2)
decrease_indices = differences <= -2
first_decrease_year = df['Year'][1:][decrease_indices].index[0] if decrease_indices.any() else None

# Get the corresponding year
if first_decrease_year is not None:
    result_year = df.loc[first_decrease_year, 'Year']
else:
    result_year = None

print(f"Final Answer: {result_year}")