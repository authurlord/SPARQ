import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Convert 'year' to integer and 'bötzow' to float
df['year'] = pd.to_numeric(df['year'])
df['bötzow'] = pd.to_numeric(df['bötzow'])

# Prepare the data for linear regression
X = df['year'].values.reshape(-1, 1)
y = df['bötzow'].values

# Fit a linear regression model
model = np.polyfit(X.flatten(), y, 1)
predicted_values = np.polyval(model, [2011, 2012])

# Output the forecasted values
print(f"Final Answer: {predicted_values[0]:.3f}, {predicted_values[1]:.3f}")