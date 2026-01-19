import pandas as pd

df = pd.read_csv('table.csv')
# Extract the year from the 'country' column by parsing the parentheses
df['year'] = df['country'].str.extract(r'\((\d{4})\)')
# Convert 'aids orphans as % of orphans' to float for comparison
df['aids_orphans_percent'] = pd.to_numeric(df['aids orphans as % of orphans'], errors='coerce')
# Find the row with the maximum percentage
max_row = df.loc[df['aids_orphans_percent'].idxmax()]
# Extract the year
final_year = max_row['year']
print(f"Final Answer: {final_year}")