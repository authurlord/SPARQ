import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for year 2005 and sum Indians and Pakistanis admitted
total_2005 = df[df['year'] == '2005']['indians admitted'].values[0] + df[df['year'] == '2005']['pakistanis admitted'].values[0]
print(f"Final Answer: {total_2005}")