import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract year from the country column (e.g., 'botswana (1990)' -> 1990)
df['year'] = df['country'].str.extract(r'\((\d{4})\)$')[0].astype(int)

# Find the row with the maximum percentage of AIDS-related orphans
max_percentage_row = df.loc[df['aids orphans as % of orphans'].idxmax()]

# Return the year associated with the maximum percentage
final_year = max_percentage_row['year']
print(f"Final Answer: {final_year}")