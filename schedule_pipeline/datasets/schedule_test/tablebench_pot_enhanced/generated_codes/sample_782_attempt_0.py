import pandas as pd

df = pd.read_csv('table.csv')
# Extract issue prices and convert to float
issue_prices = df['issue price'].astype(float)

# Observe the trend: after 2003, prices stay at 34.95 until 2007, then jump to 38.95
# Assuming a similar jump in 2008, add $4.00 to the last known price
forecasted_price = issue_prices.iloc[-1] + 4.00

print(f"Final Answer: {forecasted_price:.2f}")