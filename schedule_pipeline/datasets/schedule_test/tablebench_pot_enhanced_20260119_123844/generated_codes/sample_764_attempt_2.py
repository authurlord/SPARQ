import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'issue price' to float for calculation
df['issue price'] = pd.to_numeric(df['issue price'])
# Extract the last few years' data for trend analysis
last_years = df[df['year'].astype(int) >= 2006]
# Calculate the average annual increase from 2006 to 2011
annual_increase = last_years['issue price'].diff().mean()
# Forecast 2012 price
forecast_2012 = last_years['issue price'].iloc[-1] + annual_increase
print(f"Final Answer: {forecast_2012:.2f}")