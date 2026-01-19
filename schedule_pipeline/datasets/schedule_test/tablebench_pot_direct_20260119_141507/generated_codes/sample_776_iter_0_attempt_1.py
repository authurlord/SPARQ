import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')
# Extract years and net assets
years = [int(row[0].split('/')[0]) for row in df.values]
net_assets = df['net assets at end of year'].astype(float)

# Prepare data for linear regression: year vs net assets
X = np.array(years).reshape(-1, 1)
y = np.array(net_assets)

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict net assets for 2013 (year 2013)
predicted_asset_2013 = model.predict([[2013]])[0]

print(f"Final Answer: {predicted_asset_2013:.0f}")