import pandas as pd

df = pd.read_csv('table.csv')
# Filter for years between 1867 and 1873 (inclusive)
filtered_df = df[(df['introduced'].astype(int) >= 1867) & (df['introduced'].astype(int) <= 1873)]
# Filter for the specified types
filtered_df = filtered_df[filtered_df['type'].isin(['2 - 4 - 0', '0 - 6 - 0'])]
# Sum the 'total' column
total_locomotives = filtered_df['total'].sum()
print(f"Final Answer: {total_locomotives}")