import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv('table.csv')

# Convert year column to numeric and extract bötzow values
df['year'] = pd.to_numeric(df['year'], errors='coerce')
bötzow_values = df['bötzow'].astype(float)
years = df['year'].astype(float)

# Fit a linear regression model
model = LinearRegression()
model.fit(years.values.reshape(-1, 1), bötzow_values.values)

# Predict for next two years: 2011 and 2012
future_years = np.array([[2011], [2012]])
predicted_bötzow = model.predict(future_years)

print(f"Final Answer: {predicted_bötzow[0]:.2f}, {predicted_bötzow[1]:.2f}")