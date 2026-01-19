import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for years 2000 to 2010 (excluding the 'total' row)
filtered_df = df[df['year'].str.contains(r'^200\d$') & (df['year'] != 'total')]
# Calculate the average number of Indians admitted per year
average_indians = filtered_df['indians admitted'].mean()
print(f"Final Answer: {average_indians:.0f}")