import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Convert year to integer and issue price to float
df['year'] = pd.to_numeric(df['year'])
df['issue price'] = pd.to_numeric(df['issue price'])

# Prepare data for regression
X = df[['year']]
y = df['issue price']

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict issue price for 2012
predicted_price = model.predict([[2012]])

print(f"Final Answer: {predicted_price[0]:.2f}")