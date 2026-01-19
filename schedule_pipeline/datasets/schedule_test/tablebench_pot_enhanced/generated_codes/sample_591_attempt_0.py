import pandas as pd

df = pd.read_csv('table.csv')
# Clean the 'Live births per year' column by removing spaces and converting to integer
df['Live births per year'] = df['Live births per year'].str.replace(' ', '').astype(int)
# Sum live births from 1950-1955 to 1975-1980 (first 6 rows)
total_live_births = df['Live births per year'].iloc[:6].sum()
print(f"Final Answer: {total_live_births}")