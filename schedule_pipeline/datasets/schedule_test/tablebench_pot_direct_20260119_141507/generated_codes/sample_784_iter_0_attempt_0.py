import pandas as pd

df = pd.read_csv('table.csv')
# Convert year to numeric and extract issue price
df['year'] = pd.to_numeric(df['year'])
issue_prices = df['issue price'].astype(float)

# Calculate the trend: average increase per year
years = df['year']
prices = issue_prices
annual_increase = (prices.iloc[-1] - prices.iloc[0]) / (years.iloc[-1] - years.iloc[0])

# Forecast for 2013
forecast_2013 = prices.iloc[-1] + annual_increase
print(f"Final Answer: {forecast_2013:.2f}")