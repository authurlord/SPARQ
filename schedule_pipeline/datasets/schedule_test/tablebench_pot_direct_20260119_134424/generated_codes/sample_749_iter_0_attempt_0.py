import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Extract and convert relevant columns
df['broadcast date'] = pd.to_numeric(df['broadcast date'])
df['viewers (millions)'] = pd.to_numeric(df['viewers (millions)'])

# Prepare data for linear regression
X = df['broadcast date'].values.reshape(-1, 1)
y = df['viewers (millions)'].values

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict viewership for 2013
forecast_year = 2013
predicted_viewers = model.predict([[forecast_year]])[0]

print(f"Final Answer: {predicted_viewers:.2f}")