import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows from '1950-1955' to '1975-1980'
filtered_df = df[df['Period'].isin(['1950-1955', '1955-1960', '1960-1965', '1965-1970', '1970-1975', '1975-1980'])]
# Remove spaces in 'Live births per year' and convert to integer
filtered_df['Live births per year'] = filtered_df['Live births per year'].str.replace(' ', '').astype(int)
# Calculate total live births
total_live_births = filtered_df['Live births per year'].sum()
print(f"Final Answer: {total_live_births}")