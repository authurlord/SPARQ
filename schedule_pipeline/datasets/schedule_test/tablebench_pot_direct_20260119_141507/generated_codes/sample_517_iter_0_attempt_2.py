import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'total orphans (aids related)' to numeric (some values are strings like '< 100')
df['total orphans (aids related)'] = pd.to_numeric(df['total orphans (aids related)'], errors='coerce')

# Extract year from the country column (e.g., 'botswana (1990)' -> 1990)
df['year'] = df['country'].str.extract(r'\((\d{4})\)').astype(str).str.strip()

# Find the row with the maximum AIDS-related orphans
max_orphans_row = df.loc[df['total orphans (aids related)'].idxmax()]
max_year = max_orphans_row['year']

print(f"Final Answer: {max_year}")