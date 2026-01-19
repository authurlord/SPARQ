import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert year to numeric for regression
df['year (january)'] = pd.to_numeric(df['year (january)'])

# Extract x (year) and y (urban percentage)
x = df['year (january)']
y = df['urban , %']

# Fit a linear regression model
coefficients = np.polyfit(x, y, 1)
poly = np.poly1d(coefficients)

# Predict urban percentage for 2015
predicted_urban_2015 = poly(2015)

print(f"Final Answer: {predicted_urban_2015:.1f}")