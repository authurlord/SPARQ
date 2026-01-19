import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'year' to integer and 'Percentage (based on registration by the church itself)' to float
df['year'] = df['year'].astype(int)
df['Percentage (based on registration by the church itself)'] = df['Percentage (based on registration by the church itself)'].astype(float)

# Prepare data for linear regression
X = df['year'].values.reshape(-1, 1)
y = df['Percentage (based on registration by the church itself)'].values

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict percentage for 2025
predicted_percentage_2025 = model.predict([[2025]])[0]

print(f"Final Answer: {predicted_percentage_2025:.1f}")