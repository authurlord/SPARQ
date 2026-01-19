import pandas as pd

df = pd.read_csv('table.csv')
# Extract issue prices
issue_prices = df['issue price'].astype(float)
# Last known price
last_price = issue_prices.iloc[-1]
print(f"Final Answer: {last_price}")