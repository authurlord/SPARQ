import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'year' to integer and 'bötzow' to float for analysis
df['year'] = pd.to_numeric(df['year'])
df['bötzow'] = pd.to_numeric(df['bötzow'])

# Prepare data for linear regression
X = df['year'].values.reshape(-1, 1)
y = df['bötzow'].values

# Fit a linear regression model
model = np.polyfit(X.flatten(), y, 1)
poly = np.poly1d(model)

# Predict for 2011 and 2012
predictions = poly([2011, 2012])

print(f"Final Answer: {predictions[0]:.3f}, {predictions[1]:.3f}")