import pandas as pd

df = pd.read_csv('table.csv')
# Find the row with the maximum percentage of AIDS-related orphans
max_row = df.loc[df['aids orphans as % of orphans'].idxmax()]
# Extract the year from the country column (e.g., 'malawi (2001)')
year = max_row['country'].split('(')[-1].strip(')')
print(f"Final Answer: {year}")