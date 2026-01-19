import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'indians admitted' to integer
df['indians admitted'] = pd.to_numeric(df['indians admitted'])
# Filter rows from 2000 to 2010 (exclude 'total' row)
indians_2000_to_2010 = df[df['year'] != 'total']['indians admitted']
# Calculate average
average_indians = indians_2000_to_2010.mean()
print(f"Final Answer: {average_indians:.1f}")