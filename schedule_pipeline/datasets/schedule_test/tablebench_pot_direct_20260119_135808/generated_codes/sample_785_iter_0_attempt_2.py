import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')
# Convert percentage column to numeric
df['Percentage (based on registration by the church itself)'] = pd.to_numeric(df['Percentage (based on registration by the church itself)'])

# Prepare data for regression: year as independent variable, percentage as dependent
X = df['year'].values.reshape(-1, 1)
y = df['Percentage (based on registration by the church itself)'].values

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict for 2025
predicted_percentage = model.predict([[2025]])[0]

print(f"Final Answer: {predicted_percentage:.1f}")