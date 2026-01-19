import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for years 2000 to 2010, excluding the "total" row
df_filtered = df[df['year'].str.contains(r'^20\d{2}$')]
# Extract the 'indians admitted' column and compute the mean
avg_indians = df_filtered['indians admitted'].mean()
print(f"Final Answer: {avg_indians:.0f}")