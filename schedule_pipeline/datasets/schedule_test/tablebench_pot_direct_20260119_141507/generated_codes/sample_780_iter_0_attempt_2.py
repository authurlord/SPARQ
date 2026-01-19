import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Extract year and total support and revenue
years = [int(row[0].split('/')[0]) for row in df.values]
revenue = df['total support and revenue'].astype(float)

# Reshape the data for linear regression
X = np.array(years).reshape(-1, 1)
y = np.array(revenue).reshape(-1, 1)

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict revenue for year 2013
next_year = np.array([[2013]])
projected_revenue = model.predict(next_year)[0][0]

print(f"Final Answer: {projected_revenue:.0f}")