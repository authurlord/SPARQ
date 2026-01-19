import pandas as pd

df = pd.read_csv('table.csv')
# Extract the year from the 'country' column (e.g., 'botswana (2001)')
df['year'] = df['country'].str.extract(r'\((\d{4})\)')[0]
# Convert 'aids orphans as % of orphans' to float for comparison
df['aids_orphans_pct'] = pd.to_numeric(df['aids orphans as % of orphans'], errors='coerce')
# Find the year with the maximum percentage of AIDS-related orphans
max_year = df.loc[df['aids_orphans_pct'].idxmax(), 'year']
print(f"Final Answer: {max_year}")