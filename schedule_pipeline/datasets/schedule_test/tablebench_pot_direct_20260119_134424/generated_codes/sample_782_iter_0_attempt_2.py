import pandas as pd

df = pd.read_csv('table.csv')
# Extract issue prices
issue_prices = df['issue price'].astype(float)

# Last known price is 38.95 in 2007
# The last increase was +4.00 from 2006 to 2007
# Forecast next price by adding same increment
forecasted_price = issue_prices.iloc[-1] + 4.00
print(f"Final Answer: {forecasted_price:.2f}")