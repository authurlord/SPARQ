import pandas as pd

df = pd.read_csv('table.csv')
# Extract the year from the 'country' column (e.g., 'botswana (2001)')
df['year'] = df['country'].str.extract(r'\((\d{4})\)')[0]
# Convert 'aids orphans as % of orphans' to float for comparison
df['aids orphans as % of orphans'] = pd.to_numeric(df['aids orphans as % of orphans'], errors='coerce')
# Find the row with the maximum percentage
max_row = df.loc[df['aids orphans as % of orphans'].idxmax()]
# Extract the year
year_with_most_aids_orphans = max_row['year']
print(f"Final Answer: {year_with_most_aids_orphans}")