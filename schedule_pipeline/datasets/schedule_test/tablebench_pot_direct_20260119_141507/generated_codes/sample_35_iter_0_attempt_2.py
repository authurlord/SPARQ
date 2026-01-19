import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where type is '2 - 4 - 0' or '0 - 6 - 0' and introduced between 1867 and 1873
filtered_df = df[
    (df['type'].str.contains('2 - 4 - 0') | df['type'].str.contains('0 - 6 - 0')) &
    (df['introduced'].astype(int) >= 1867) &
    (df['introduced'].astype(int) <= 1873)
]

# Sum the 'total' column of filtered rows
total_locomotives = filtered_df['total'].sum()

print(f"Final Answer: {total_locomotives}")