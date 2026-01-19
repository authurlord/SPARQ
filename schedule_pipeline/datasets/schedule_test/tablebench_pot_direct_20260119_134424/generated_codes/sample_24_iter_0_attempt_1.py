import pandas as pd

df = pd.read_csv('table.csv')
# Filter data from 2000 to 2008
df_filtered = df[df['year'].astype(int).between(2000, 2008)]
# Sum all admission columns
total_admitted = df_filtered[['indians admitted', 'pakistanis admitted', 'sri lankans admitted', 'bangladeshis admitted', 'nepalis admitted']].sum().sum()
print(f"Final Answer: {total_admitted}")