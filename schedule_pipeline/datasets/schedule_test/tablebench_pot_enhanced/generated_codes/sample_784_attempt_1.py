import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert year to numeric and issue price to float
df['year'] = pd.to_numeric(df['year'])
df['issue price'] = pd.to_numeric(df['issue price'])

# Prepare data for linear regression
X = df['year'].values.reshape(-1, 1)
y = df['issue price'].values

# Fit linear regression model
model = np.polyfit(X.flatten(), y, 1)
predicted_price = np.polyval(model, 2013)

print(f"Final Answer: {predicted_price:.2f}")