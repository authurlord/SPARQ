import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'Year' to integer for proper comparison
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Sort by Year to ensure chronological order
df = df.sort_values('Year').reset_index(drop=True)

# Extract Conservative councillors values
conservative_councillors = df['Conservative councillors']

# Compute the difference from previous year
diffs = conservative_councillors.diff()

# Find the index of the maximum increase (excluding the first row, which has no previous)
max_increase_idx = diffs.idxmax() if diffs.max() > 0 else None

# Get the corresponding year
if max_increase_idx is not None:
    year_with_max_increase = df.loc[max_increase_idx, 'Year']
else:
    year_with_max_increase = None

print(f"Final Answer: {year_with_max_increase}")