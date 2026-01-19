import pandas as pd

df = pd.read_csv('table.csv')
# Convert issue_price to float
df['issue price'] = pd.to_numeric(df['issue price'])

# Extract the years and issue prices
years = df['year'].astype(int)
prices = df['issue price']

# Calculate the average annual increase in issue price
avg_increase = prices.diff().mean()

# Forecast the price for 2012
forecast_2012 = prices.iloc[-1] + avg_increase

print(f"Final Answer: {forecast_2012:.2f}")