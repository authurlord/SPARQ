import pandas as pd

df = pd.read_csv('table.csv')
# Extract issue prices and analyze trend
issue_prices = df['issue price'].astype(float)
# Last known price
last_price = issue_prices.iloc[-1]
# Last increase was $4
increase = 4.00
# Forecast next price
forecasted_price = last_price + increase
print(f"Final Answer: {forecasted_price:.2f}")