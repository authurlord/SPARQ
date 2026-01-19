import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'total orphans (aids related)' to numeric, handling '< 100' as 0 for comparison
df['total orphans (aids related)'] = pd.to_numeric(df['total orphans (aids related)'], errors='coerce').fillna(0)

# Find the row with the maximum number of AIDS-related orphans
max_row = df.loc[df['total orphans (aids related)'].idxmax()]

# Extract the year from the 'country' column (e.g., 'uganda (2001)')
year = max_row['country'].split('(')[-1].strip(')')

print(f"Final Answer: {year}")