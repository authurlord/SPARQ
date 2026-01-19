import pandas as pd

df = pd.read_csv('table.csv')
# Filter by introduced year between 1867 and 1873 inclusive
filtered_df = df[(df['introduced'].astype(int) >= 1867) & (df['introduced'].astype(int) <= 1873)]
# Filter by type '2 - 4 - 0' or '0 - 6 - 0'
filtered_df = filtered_df[filtered_df['type'].isin(['2 - 4 - 0', '0 - 6 - 0'])]
# Sum the total number of locomotives
total_locomotives = filtered_df['total'].sum()
print(f"Final Answer: {total_locomotives}")