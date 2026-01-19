import pandas as pd

df = pd.read_csv('table.csv')
# Extract the issue price for 2011
last_price = df[df['year'] == '2011']['issue price'].values[0]
print(f"Final Answer: {last_price}")