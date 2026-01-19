import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Convert year column to numeric
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Prepare data for regression
X_year = df[['year']]
y_catholics = df['Catholics (based on registration by the church itself)']
y_percentage = df['Percentage (based on registration by the church itself)']

# Fit linear regression models
model_catholics = LinearRegression()
model_percentage = LinearRegression()

model_catholics.fit(X_year, y_catholics)
model_percentage.fit(X_year, y_percentage)

# Predict for year 2025
year_2025 = 2025
catholics_2025 = model_catholics.predict([[year_2025]])[0]
percentage_2025 = model_percentage.predict([[year_2025]])[0]

print(f"Final Answer: {catholics_2025:.0f}, {percentage_2025:.1f}")