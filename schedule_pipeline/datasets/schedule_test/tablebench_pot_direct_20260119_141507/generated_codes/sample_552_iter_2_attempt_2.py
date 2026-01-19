import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for years 2000 to 2010 and exclude the 'total' row
filtered_df = df[df['year'].str.contains(r'^20\d{2}$') & (df['year'] != 'total')]
# Calculate the average number of Indians admitted per year
average_indians = filtered_df['indians admitted'].mean()
print(f"Final Answer: {average_indians:.0f}")