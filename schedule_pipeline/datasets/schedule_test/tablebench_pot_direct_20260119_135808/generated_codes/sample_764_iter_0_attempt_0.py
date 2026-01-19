import pandas as pd

df = pd.read_csv('table.csv')
# Convert year and issue price to numeric
df['year'] = pd.to_numeric(df['year'])
df['issue price'] = pd.to_numeric(df['issue price'])

# Calculate the average annual increase in issue price
price_diff = df['issue price'].diff().dropna()
avg_increase = price_diff.mean()

# Forecast 2012 price by adding the average increase to 2011 price
forecast_2012 = df.loc[df['year'] == 2011, 'issue price'].values[0] + avg_increase

print(f"Final Answer: {forecast_2012:.2f}")