import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the 'total' row and keep only years from 2000 to 2010
filtered_df = df[df['year'].str.contains(r'^20\d{2}$') & (df['year'] != 'total')]
# Calculate the average number of Indians admitted
average_indians = filtered_df['indians admitted'].mean()
print(f"Final Answer: {average_indians:.0f}")