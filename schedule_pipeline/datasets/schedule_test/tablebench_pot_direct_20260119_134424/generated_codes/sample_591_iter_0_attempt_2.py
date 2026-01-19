import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows from 1950-1955 to 1975-1980
filtered_df = df[(df['Period'] >= '1950-1955') & (df['Period'] <= '1975-1980')]
# Remove spaces in 'Live births per year' and convert to integer
live_births = filtered_df['Live births per year'].str.replace(' ', '').astype(int)
total_live_births = live_births.sum()
print(f"Final Answer: {total_live_births}")