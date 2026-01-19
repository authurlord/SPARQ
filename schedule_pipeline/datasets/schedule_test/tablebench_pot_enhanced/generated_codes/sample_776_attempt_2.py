import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Extract the year part (e.g., 2003 from "2003 / 2004") and convert to integer
df['year_numeric'] = df['year'].str.extract(r'(\d{4})').astype(int)

# Prepare the data for regression
X = df['year_numeric'].values.reshape(-1, 1)
y = df['net assets at end of year'].values

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict for 2013 (next year after 2012)
predicted_net_assets = model.predict([[2013]])[0]

# Round to nearest whole number
print(f"Final Answer: {int(round(predicted_net_assets))}")