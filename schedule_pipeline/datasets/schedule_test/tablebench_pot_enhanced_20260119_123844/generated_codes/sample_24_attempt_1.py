import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 2000 to 2008
df_filtered = df[df['year'].astype(int) >= 2000][df['year'].astype(int) <= 2008]
# Convert admission columns to numeric and sum them
total_admitted = df_filtered[['indians admitted', 'pakistanis admitted', 'sri lankans admitted', 'bangladeshis admitted', 'nepalis admitted']].sum().sum()
print(f"Final Answer: {total_admitted}")