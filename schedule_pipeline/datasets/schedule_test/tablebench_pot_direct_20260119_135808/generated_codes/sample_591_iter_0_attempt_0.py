import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for the required periods
filtered_df = df[(df['Period'] >= '1950-1955') & (df['Period'] <= '1975-1980')]
# Convert 'Live births per year' to integer after removing spaces
filtered_df['Live births per year'] = filtered_df['Live births per year'].str.replace(' ', '').astype(int)
total_live_births = filtered_df['Live births per year'].sum()
print(f"Final Answer: {total_live_births}")