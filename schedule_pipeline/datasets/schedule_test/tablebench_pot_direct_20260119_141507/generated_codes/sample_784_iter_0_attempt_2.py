import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert year to numeric and extract issue price
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['issue price'] = df['issue price'].str.replace('$', '').astype(float)

# Prepare data for linear regression
years = df['year'].values
prices = df['issue price'].values

# Fit a linear regression model: price = a * year + b
coeffs = np.polyfit(years, prices, 1)
forecast_price = coeffs[0] * 2013 + coeffs[1]

print(f"Final Answer: {forecast_price:.2f}")