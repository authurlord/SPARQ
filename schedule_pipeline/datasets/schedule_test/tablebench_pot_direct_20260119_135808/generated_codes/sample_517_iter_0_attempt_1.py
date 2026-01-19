import pandas as pd

df = pd.read_csv('table.csv')
# Find the row with the maximum percentage of AIDS orphans
max_row = df.loc[df['aids orphans as % of orphans'].idxmax()]
# Extract the year from the country column (e.g., 'botswana (1990)')
year = max_row['country'].split('(')[1].split(')')[0]
print(f"Final Answer: {year}")