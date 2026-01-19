import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Convert year to numeric and extract percentage
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['percentage'] = pd.to_numeric(df['Percentage (based on registration by the church itself)'], errors='coerce')

# Prepare data for regression
X = df[['year']].values.reshape(-1, 1)
y = df['percentage'].values

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict percentage for 2025
predicted_percentage_2025 = model.predict([[2025]])[0]

print(f"Final Answer: {predicted_percentage_2025:.1f}")