import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where country contains "Democratic Republic of the Congo" and elevation >= 3000
filtered_df = df[(df['country'].str.contains('Democratic Republic of the congo', case=False)) & (df['elevation (m)'].astype(int) >= 3000)]
# Calculate average prominence of the filtered rows
average_prominence = filtered_df['prominence (m)'].mean()
print(f"Final Answer: {average_prominence:.1f}")