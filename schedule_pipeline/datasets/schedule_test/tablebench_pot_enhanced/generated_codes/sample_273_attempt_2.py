import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where country is 'democratic republic of the congo' and elevation >= 3000
filtered_df = df[(df['country'] == 'democratic republic of the congo') & (df['elevation (m)'].astype(int) >= 3000)]
# Calculate average prominence
avg_prominence = filtered_df['prominence (m)'].astype(int).mean()
print(f"Final Answer: {avg_prominence:.1f}")