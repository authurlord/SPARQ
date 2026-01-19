import pandas as pd

df = pd.read_csv('table.csv')

# Filter for the specified years and types
filtered_df = df[
    (df['introduced'].astype(int) >= 1867) &
    (df['introduced'].astype(int) <= 1873) &
    (df['type'].isin(['2 - 4 - 0', '0 - 6 - 0']))
]

# Sum the total number of locomotives
total_locomotives = filtered_df['total'].sum()

print(f"Final Answer: {total_locomotives}")