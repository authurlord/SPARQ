import pandas as pd

df = pd.read_csv('table.csv')
# Extract the issue price and year
issue_prices = df['issue price'].astype(float)
years = df['year'].astype(int)

# Calculate the average annual increase in issue price
annual_increase = (issue_prices.iloc[-1] - issue_prices.iloc[0]) / (years.iloc[-1] - years.iloc[0])

# Forecast the issue price for 2008
forecasted_price = issue_prices.iloc[-1] + annual_increase

print(f"Final Answer: {forecasted_price:.2f}")