import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'all bills sponsored' to integer
df['all bills sponsored'] = df['all bills sponsored'].astype(int)
# Filter rows where 'all bills sponsored' > 50
filtered_df = df[df['all bills sponsored'] > 50]
# Sum the 'bills originally cosponsored' for those rows
total_cosponsored = filtered_df['bills originally cosponsored'].sum()
print(f"Final Answer: {total_cosponsored}")