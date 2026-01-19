import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Extract the relevant columns: Year_2 and its corresponding value
years = df['Year_2'].astype(int)
values = df['-_2'].str.replace(',', '').astype(float)

# Reshape for sklearn
X = years.values.reshape(-1, 1)
y = values.values

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict for year 2020
prediction = model.predict([[2020]])

print(f"Final Answer: {prediction[0]:.0f}")