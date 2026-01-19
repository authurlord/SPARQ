import pandas as pd

df = pd.read_csv('table.csv')
# Filter data from 1950-1955 to 1975-1980
filtered_df = df[(df['Period'] >= '1950-1955') & (df['Period'] <= '1975-1980')]
# Clean 'Live births per year' by removing spaces and converting to int
filtered_df['Live births per year'] = filtered_df['Live births per year'].str.replace(' ', '').astype(int)
# Calculate total live births
total_live_births = filtered_df['Live births per year'].sum()
print(f"Final Answer: {total_live_births}")