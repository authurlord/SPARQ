import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')
# Extract years and total steel production
years = np.array([2000, 2001, 2002, 2003, 2004, 2005, 2006])
total_s_ton = np.array([3157247, 2861134, 2983137, 3178633, 3085753, 3527469, 2926536])

# Reshape for linear regression
X = years.reshape(-1, 1)
y = total_s_ton

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict for 2007
forecast_2007 = model.predict([[2007]])[0]
print(f"Final Answer: {forecast_2007:.0f}")