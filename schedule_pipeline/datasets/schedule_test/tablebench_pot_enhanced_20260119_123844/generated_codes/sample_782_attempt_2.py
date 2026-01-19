import pandas as pd

df = pd.read_csv('table.csv')

# Extract the issue prices
issue_prices = df['issue price'].astype(float)

# Last known price
last_price = issue_prices.iloc[-1]

# Calculate the increase from 2006 to 2007
increase = last_price - issue_prices.iloc[-2]

# Forecast next price
forecasted_price = last_price + increase

print(f"Final Answer: {forecasted_price:.2f}")