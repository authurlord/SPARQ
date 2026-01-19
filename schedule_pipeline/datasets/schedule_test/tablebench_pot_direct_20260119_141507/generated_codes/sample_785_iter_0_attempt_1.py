import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Convert year to numeric
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Prepare features and target
X = df[['year']]
y = df['Percentage (based on registration by the church itself)']

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict percentage for year 2025
predicted_percentage_2025 = model.predict([[2025]])[0]

print(f"Final Answer: {predicted_percentage_2025:.1f}")