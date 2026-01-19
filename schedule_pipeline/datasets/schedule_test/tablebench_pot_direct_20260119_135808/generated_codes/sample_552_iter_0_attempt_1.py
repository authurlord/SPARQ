import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'indians admitted' to integer, excluding the 'total' row
indians_admitted = df['indians admitted'].iloc[:-1].astype(int)
# Calculate average
average_indians = indians_admitted.mean()
print(f"Final Answer: {average_indians:.1f}")