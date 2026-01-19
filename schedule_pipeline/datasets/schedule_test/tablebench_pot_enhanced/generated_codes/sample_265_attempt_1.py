import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'all bills sponsored' > 50
filtered_df = df[df['all bills sponsored'] > 50]
# Sum 'bills originally cosponsored' for these rows
total_cosponsored = filtered_df['bills originally cosponsored'].sum()
print(f"Final Answer: {total_cosponsored}")