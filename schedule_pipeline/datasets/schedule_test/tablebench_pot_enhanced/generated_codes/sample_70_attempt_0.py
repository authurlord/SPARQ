import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where year is 2005
indians_2005 = df[df['year'] == '2005']['indians admitted'].iloc[0]
print(f"Final Answer: {indians_2005}")