import pandas as pd

df = pd.read_csv('table.csv')

# Filter for types '2 - 4 - 0' or '0 - 6 - 0'
filtered_df = df[(df['type'].str.contains('2 - 4 - 0') | df['type'].str.contains('0 - 6 - 0'))]

# Further filter introduced between 1867 and 1873 (inclusive)
filtered_df = filtered_df[(filtered_df['introduced'] >= 1867) & (filtered_df['introduced'] <= 1873)]

# Sum the 'total' column
total_locomotives = filtered_df['total'].sum()

print(f"Final Answer: {total_locomotives}")