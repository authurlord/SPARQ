import pandas as pd

df = pd.read_csv('table.csv')
# Find the row where year is 2005 and get the value for 'indians admitted'
indians_2005 = df[df['year'] == '2005']['indians admitted'].values[0]
print(f"Final Answer: {indians_2005}")