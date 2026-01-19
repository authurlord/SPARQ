import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'indians admitted' column to integers
df['indians admitted'] = df['indians admitted'].astype(int)
# Exclude the 'total' row and calculate average for years 2000 to 2010
average_indians = df.loc[0:10, 'indians admitted'].mean()
print(f"Final Answer: {average_indians:.1f}")