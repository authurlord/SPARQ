import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'issue price' to float for calculation
df['issue price'] = df['issue price'].astype(float)

# Calculate the annual increase in issue price
years = df['year'].astype(int)
prices = df['issue price']
annual_increase = (prices.iloc[-1] - prices.iloc[0]) / (years.iloc[-1] - years.iloc[0])

# Forecast for 2008
forecasted_price = prices.iloc[-1] + annual_increase
print(f"Final Answer: {forecasted_price:.2f}")