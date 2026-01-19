import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv('table.csv')

# Clean the data
df['population'] = df['population'].str.replace(',', '').astype(int)
df['Catholics (based on registration by the church itself)'] = df['Catholics (based on registration by the church itself)'].str.replace(',', '').astype(int)
df['Percentage (based on registration by the church itself)'] = df['Percentage (based on registration by the church itself)'].astype(float)

# Prepare features (year) and target variables
X = df['year'].values.reshape(-1, 1)
y_catholics = df['Catholics (based on registration by the church itself)'].values
y_percentage = df['Percentage (based on registration by the church itself)'].values

# Fit linear regression models
model_catholics = LinearRegression().fit(X, y_catholics)
model_percentage = LinearRegression().fit(X, y_percentage)

# Predict for 2025
X_2025 = np.array([[2025]])
predicted_catholics = model_catholics.predict(X_2025)[0]
predicted_percentage = model_percentage.predict(X_2025)[0]

# Round to nearest whole number for Catholics and one decimal for percentage
predicted_catholics = round(predicted_catholics)
predicted_percentage = round(predicted_percentage, 1)

print(f"Final Answer: {predicted_catholics}, {predicted_percentage}")