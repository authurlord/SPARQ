import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')
# Extract year and issue price
years = df['year'].astype(int)
prices = df['issue price'].astype(float)

# Prepare data for linear regression: year as feature, price as target
X = years.values.reshape(-1, 1)
y = prices.values

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict price for 2012
forecast_2012 = model.predict([[2012]])[0]
print(f"Final Answer: {forecast_2012:.2f}")