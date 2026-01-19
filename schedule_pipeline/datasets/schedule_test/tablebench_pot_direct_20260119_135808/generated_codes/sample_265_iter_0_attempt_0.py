import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'all bills sponsored' > 50
filtered_df = df[df['all bills sponsored'] > 50]
# Sum the 'bills originally cosponsored' column
total_bills_cosponsored = filtered_df['bills originally cosponsored'].sum()
print(f"Final Answer: {total_bills_cosponsored}")