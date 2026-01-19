import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')
# Prepare the data: years as index, net assets at end of year as target
years = [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011]
net_assets = df['net assets at end of year'].astype(float)

# Create a feature array for years
X = np.array(years).reshape(-1, 1)
y = np.array(net_assets)

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict net assets for 2012 (corresponding to 2012/2013)
predicted_2012 = model.predict([[2012]])[0]

print(f"Final Answer: {predicted_2012:.0f}")