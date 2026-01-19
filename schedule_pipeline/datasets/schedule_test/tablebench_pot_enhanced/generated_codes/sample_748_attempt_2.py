import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert year to numeric and prepare data for regression
df['year'] = pd.to_numeric(df['year'])
X = df['year'].values.reshape(-1, 1)  # Independent variable: year
y = df['qual'].astype(float).values  # Dependent variable: qualifying speed

# Fit a linear regression model
model = np.polyfit(X.flatten(), y, 1)
predicted_speed_1960 = np.polyval(model, 1960)

print(f"Final Answer: {predicted_speed_1960:.3f}")