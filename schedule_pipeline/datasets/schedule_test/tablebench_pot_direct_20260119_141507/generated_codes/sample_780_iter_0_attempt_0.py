import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Convert year to numeric (extract from the 'year' column)
df['year'] = df['year'].str.replace('/','').astype(int)

# Prepare data: years as independent variable, total support and revenue as dependent
X = df[['year']]
y = df['total support and revenue']

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict for the next year: 2012/2013 → year 2013
next_year = 2013
predicted_revenue = model.predict([[next_year]])[0]

print(f"Final Answer: {predicted_revenue:.0f}")