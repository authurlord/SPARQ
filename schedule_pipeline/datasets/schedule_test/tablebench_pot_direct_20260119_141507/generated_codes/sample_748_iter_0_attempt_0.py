import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')
# Prepare data: years as x (1951 to 1957), qual as y
years = np.array([int(row[0]) for row in df.values]) - 1950  # Convert to offset from 1950
qual_speeds = np.array([float(row[2]) for row in df.values])

# Reshape for sklearn
X = years.reshape(-1, 1)
y = qual_speeds

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict qualifying speed for year 1960 (offset = 10)
predicted_qual_1960 = model.predict([[10]])[0]

print(f"Final Answer: {predicted_qual_1960:.2f}")