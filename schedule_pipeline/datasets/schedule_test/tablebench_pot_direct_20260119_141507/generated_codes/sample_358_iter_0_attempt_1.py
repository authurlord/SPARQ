import pandas as pd

df = pd.read_csv('table.csv')
# Count years where 'indians admitted' > 25000
count_years = df[df['indians admitted'] > 25000].shape[0]
print(f"Final Answer: {count_years}")