import pandas as pd

df = pd.read_csv('table.csv')
# Extract the issue price column and filter the last three years (2009, 2010, 2011)
last_three_prices = df[df['year'].isin(['2009', '2010', '2011'])]['issue price'].astype(float)
average_last_three = last_three_prices.mean()
# Forecast 2012 by adding a small increment (e.g., 10) to account for slight upward trend
forecast_2012 = average_last_three + 10
print(f"Final Answer: {forecast_2012:.2f}")