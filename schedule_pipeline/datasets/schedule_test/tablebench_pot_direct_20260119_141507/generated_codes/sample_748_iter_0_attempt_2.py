import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Prepare data: year as feature, qual as target
years = df['year'].astype(int)
qual_speeds = df['qual'].astype(float)

# Reshape for linear regression
X = years.values.reshape(-1, 1)
y = qual_speeds.values

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict qualifying speed for 1960
predicted_qual_1960 = model.predict([[1960]])[0]

print(f"Final Answer: {predicted_qual_1960:.3f}")