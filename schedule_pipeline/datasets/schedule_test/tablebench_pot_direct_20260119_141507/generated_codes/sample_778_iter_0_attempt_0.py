import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Extract year and value
years = df['Year'].astype(int)
values = df.iloc[:, 1].str.replace(',', '').astype(int)

# Reshape for sklearn
X = years.values.reshape(-1, 1)
y = values.values

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict for year 2020
predicted_value = model.predict([[2020]])[0]
print(f"Final Answer: {predicted_value:.0f}")