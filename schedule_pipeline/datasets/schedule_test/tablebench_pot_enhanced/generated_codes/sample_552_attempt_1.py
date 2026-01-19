import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'indians admitted' to integer
df['indians admitted'] = pd.to_numeric(df['indians admitted'])
# Exclude the 'total' row and calculate mean for 2000-2010
mean_indians = df.loc[0:10, 'indians admitted'].mean()
print(f"Final Answer: {mean_indians:.1f}")