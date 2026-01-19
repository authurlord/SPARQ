import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract year from the 'country' column (e.g., 'botswana (1990)' -> 1990)
df['year'] = df['country'].str.extract(r'\((\d{4})\)$')[0].astype(int)

# Select the column for AIDS-related orphans
aids_orphans = df['total orphans (aids related)']

# Find the year with the maximum AIDS-related orphans
max_year = df.loc[aids_orphans.idxmax(), 'year']
print(f"Final Answer: {max_year}")