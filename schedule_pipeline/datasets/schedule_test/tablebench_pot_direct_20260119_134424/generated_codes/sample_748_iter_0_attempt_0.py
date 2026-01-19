import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert year to numeric and use it as the independent variable
X = df['year'].astype(int).values.reshape(-1, 1)
y = df['qual'].astype(float).values

# Fit a linear regression model
model = np.polyfit(X.flatten(), y, 1)
predicted_speed = np.polyval(model, 1960)

print(f"Final Answer: {predicted_speed:.3f}")