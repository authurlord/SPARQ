import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'indians admitted' to integer (just in case)
df['indians admitted'] = pd.to_numeric(df['indians admitted'], errors='coerce')
# Filter years where indians admitted > 25000
count = (df['indians admitted'] > 25000).sum()
print(f"Final Answer: {count}")