import pandas as pd

df = pd.read_csv('table.csv')
# Extract issue prices and years
issue_prices = df['issue price'].astype(float)
years = df['year'].astype(int)

# Calculate the differences in issue prices between consecutive years
price_diffs = issue_prices.diff().dropna()

# Last known issue price
last_price = issue_prices.iloc[-1]

# Forecast next price: assume similar increment as last one
forecast_increase = price_diffs.iloc[-1]
forecasted_price = last_price + forecast_increase

print(f"Final Answer: {forecasted_price:.2f}")