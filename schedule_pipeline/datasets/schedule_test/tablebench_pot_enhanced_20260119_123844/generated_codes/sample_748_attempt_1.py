import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert year to numeric and ensure qual is float
df['year'] = pd.to_numeric(df['year'])
df['qual'] = pd.to_numeric(df['qual'])

# Perform linear regression
X = df['year'].values.reshape(-1, 1)
y = df['qual'].values

# Fit the model
model = np.polyfit(X.flatten(), y, 1)
predicted_speed = np.polyval(model, 1960)

print(f"Final Answer: {predicted_speed:.3f}")