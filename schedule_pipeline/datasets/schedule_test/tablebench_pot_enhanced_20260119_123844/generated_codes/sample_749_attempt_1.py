import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Extract relevant columns and convert to numeric
df['broadcast date'] = pd.to_numeric(df['broadcast date'])
df['viewers (millions)'] = pd.to_numeric(df['viewers (millions)'])

# Prepare data for linear regression
X = df['broadcast date'].values.reshape(-1, 1)
y = df['viewers (millions)'].values

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict for 2013
predicted_viewers = model.predict([[2013]])

print(f"Final Answer: {predicted_viewers[0]:.2f}")