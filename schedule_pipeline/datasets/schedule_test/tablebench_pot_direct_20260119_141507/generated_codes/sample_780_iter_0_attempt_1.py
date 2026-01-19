import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')
# Convert year to numeric (e.g., 2003, 2004, etc.)
df['year'] = df['year'].str.replace('/','').astype(int)

# Extract features and target
X = df[['year']]  # independent variable
y = df['total support and revenue']  # dependent variable

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict for the next year (2012)
next_year = 2012
predicted_revenue = model.predict([[next_year]])[0]

print(f"Final Answer: {predicted_revenue:.0f}")