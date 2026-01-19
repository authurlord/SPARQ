import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Convert 'total support and revenue' to numeric, removing commas if any
df['total support and revenue'] = df['total support and revenue'].str.replace(',', '').astype(int)

# Create year index (e.g., 2003, 2004, etc.) for modeling
df['year_num'] = df['year'].str.extract(r'(\d{4})').astype(int)

# Prepare data for linear regression
X = df['year_num'].values.reshape(-1, 1)
y = df['total support and revenue'].values

# Fit linear model
model = LinearRegression()
model.fit(X, y)

# Predict for the next year (2012/2013 → 2013)
next_year = 2013
predicted_revenue = model.predict([[next_year]])[0]

print(f"Final Answer: {int(predicted_revenue)}")