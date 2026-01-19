import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Extract years and urban percentage
years = df['year (january)'].astype(int)
urban_percent = df['urban , %'].astype(float)

# Prepare data for regression: year as feature, urban % as target
X = years.values.reshape(-1, 1)
y = urban_percent.values

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict urban percentage for 2015
predicted_urban_2015 = model.predict([[2015]])[0]

print(f"Final Answer: {predicted_urban_2015:.1f}")