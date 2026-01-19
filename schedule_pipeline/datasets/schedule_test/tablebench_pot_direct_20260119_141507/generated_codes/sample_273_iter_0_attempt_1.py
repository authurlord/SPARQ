import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where country contains 'democratic republic of the congo' and elevation >= 3000
filtered_df = df[(df['country'].str.contains('democratic republic of the congo', case=False)) & (df['elevation (m)'].astype(int) >= 3000)]
# Calculate the mean of prominence for the filtered rows
average_prominence = filtered_df['prominence (m)'].mean()
print(f"Final Answer: {average_prominence:.1f}")