import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv('table.csv')

# Extract relevant data: Year_2 (years) and -_2 (values)
years = df['Year_2'].astype(int)
values = df['-_2'].str.replace(',', '').astype(int)

# Reshape for sklearn
X = years.values.reshape(-1, 1)
y = values.values

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict for 2020
predicted_2020 = model.predict([[2020]])[0]

print(f"Final Answer: {predicted_2020:.0f}")