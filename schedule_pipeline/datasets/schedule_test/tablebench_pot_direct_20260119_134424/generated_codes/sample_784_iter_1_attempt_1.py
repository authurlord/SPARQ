import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')
# Convert 'issue price' to float
df['issue price'] = df['issue price'].astype(float)
# Prepare data for regression: year as X, issue price as y
X = df['year'].values.reshape(-1, 1)
y = df['issue price'].values
# Fit linear regression model
model = LinearRegression()
model.fit(X, y)
# Predict for 2013
predicted_price = model.predict([[2013]])
print(f"Final Answer: {predicted_price[0]:.2f}")