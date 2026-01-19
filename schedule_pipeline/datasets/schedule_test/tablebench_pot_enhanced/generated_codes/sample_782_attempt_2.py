import pandas as pd

df = pd.read_csv('table.csv')
# Extract the issue price for 2007
last_price = df['issue price'].iloc[-1]
print(f"Final Answer: {last_price}")