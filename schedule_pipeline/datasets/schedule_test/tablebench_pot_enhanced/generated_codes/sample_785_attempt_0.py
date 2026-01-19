import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv('table.csv')

# Convert year to numeric and percentage to float
df['year'] = pd.to_numeric(df['year'])
df['Percentage (based on registration by the church itself)'] = pd.to_numeric(df['Percentage (based on registration by the church itself)'])

# Prepare data for regression
X = df['year'].values.reshape(-1, 1)
y = df['Percentage (based on registration by the church itself)'].values

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict for 2025
prediction_2025 = model.predict([[2025]])[0]

print(f"Final Answer: {prediction_2025:.1f}")