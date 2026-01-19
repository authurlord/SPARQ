import pandas as pd

df = pd.read_csv('table.csv')
# Extract the year from the 'country' column
df['year'] = df['country'].str.extract(r'(\d{4})')
# Convert 'aids orphans as % of orphans' to float
df['aids orphans as % of orphans'] = pd.to_numeric(df['aids orphans as % of orphans'], errors='coerce')
# Find the row with the maximum percentage
max_row = df.loc[df['aids orphans as % of orphans'].idxmax()]
# Extract the year
year = max_row['year']
print(f"Final Answer: {year}")