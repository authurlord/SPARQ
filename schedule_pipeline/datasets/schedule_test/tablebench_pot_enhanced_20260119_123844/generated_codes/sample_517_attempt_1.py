import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'aids orphans as % of orphans' to float for comparison
df['aids orphans as % of orphans'] = pd.to_numeric(df['aids orphans as % of orphans'], errors='coerce')
# Find the row with the maximum percentage
max_row = df.loc[df['aids orphans as % of orphans'].idxmax()]
# Extract the year from the country column
year = max_row['country'].split('(')[1].split(')')[0]
print(f"Final Answer: {year}")