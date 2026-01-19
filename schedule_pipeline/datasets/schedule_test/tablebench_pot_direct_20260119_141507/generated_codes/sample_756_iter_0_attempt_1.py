import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Extract columns
X = df[['registered voters']]  # independent variable: registered voters
y = df['total candidates']     # dependent variable: total candidates

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict total candidates for 500,000 registered voters
predicted_candidates = model.predict([[500000]])[0]

print(f"Final Answer: {predicted_candidates:.0f}")