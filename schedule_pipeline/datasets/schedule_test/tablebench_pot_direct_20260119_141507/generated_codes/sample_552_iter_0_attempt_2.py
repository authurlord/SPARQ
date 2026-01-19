import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for years 2000 to 2010 (exclude 'total' row)
df_filtered = df[df['year'].str.startswith('200') & (df['year'] != 'total')]
# Calculate the average of 'indians admitted'
avg_indians = df_filtered['indians admitted'].mean()
print(f"Final Answer: {avg_indians:.0f}")