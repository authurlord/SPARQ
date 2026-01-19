import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'all bills sponsored' to integer
df['all bills sponsored'] = pd.to_numeric(df['all bills sponsored'])
# Filter rows where 'all bills sponsored' > 50
filtered_df = df[df['all bills sponsored'] > 50]
# Sum 'bills originally cosponsored' for those rows
total_cosponsored = filtered_df['bills originally cosponsored'].sum()
print(f"Final Answer: {total_cosponsored}")