import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv('table.csv')

# Convert 'issue price' to float, replacing 'n/a' with NaN and dropping rows with missing values
df['issue price'] = pd.to_numeric(df['issue price'], errors='coerce')
df.dropna(subset=['issue price'], inplace=True)

# Prepare the data for linear regression
X = df['year'].values.reshape(-1, 1)  # Independent variable: year
y = df['issue price'].values           # Dependent variable: issue price

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict the issue price for 2013
predicted_price = model.predict([[2013]])[0]

print(f"Final Answer: {predicted_price:.2f}")