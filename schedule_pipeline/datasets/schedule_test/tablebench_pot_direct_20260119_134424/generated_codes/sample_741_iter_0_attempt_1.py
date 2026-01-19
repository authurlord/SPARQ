import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv('table.csv')

# Convert columns to numeric
df['indians admitted'] = pd.to_numeric(df['indians admitted'])
df['bangladeshis admitted'] = pd.to_numeric(df['bangladeshis admitted'])

# Prepare data for regression
X = df['year'].values.reshape(-1, 1)
y_indians = df['indians admitted'].values
y_bangladeshis = df['bangladeshis admitted'].values

# Fit linear regression models
model_indians = LinearRegression().fit(X, y_indians)
model_bangladeshis = LinearRegression().fit(X, y_bangladeshis)

# Predict for 2013
prediction_indians = model_indians.predict([[2013]])[0]
prediction_bangladeshis = model_bangladeshis.predict([[2013]])[0]

# Round predictions to nearest integer
predicted_indians = round(prediction_indians)
predicted_bangladeshis = round(prediction_bangladeshis)

print(f"Final Answer: {predicted_indians}, {predicted_bangladeshis}")