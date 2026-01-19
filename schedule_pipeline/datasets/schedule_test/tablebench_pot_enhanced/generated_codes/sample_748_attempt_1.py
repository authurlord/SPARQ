import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')
# Convert 'year' and 'qual' to appropriate types
df['year'] = pd.to_numeric(df['year'])
df['qual'] = pd.to_numeric(df['qual'])

# Prepare data for regression
X = df[['year']]
y = df['qual']

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict for year 1960
prediction = model.predict([[1960]])
print(f"Final Answer: {prediction[0]:.3f}")