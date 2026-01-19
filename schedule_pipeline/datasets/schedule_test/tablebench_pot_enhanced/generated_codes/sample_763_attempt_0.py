import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'year' to integer and 'bötzow' to float
df['year'] = pd.to_numeric(df['year'])
df['bötzow'] = pd.to_numeric(df['bötzow'])

# Prepare data for linear regression
X = df['year'].values.reshape(-1, 1)
y = df['bötzow'].values

# Fit a linear regression model
model = np.polyfit(X.flatten(), y, 1)
predicted_2011 = np.polyval(model, 2011)
predicted_2012 = np.polyval(model, 2012)

print(f"Final Answer: {predicted_2011:.3f}, {predicted_2012:.3f}")