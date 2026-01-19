import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')
# Prepare data: years as x, qualifying speeds as y
years = np.array([int(row[0]) for row in df.values])  # Convert year string to int
qual_speeds = np.array([float(row[1]) for row in df.values])

# Reshape years and speeds for linear regression
X = years.reshape(-1, 1)
y = qual_speeds.reshape(-1, 1)

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict qualifying speed for 1960
predicted_qual_speed = model.predict([[1960]])[0][0]
print(f"Final Answer: {predicted_qual_speed:.2f}")