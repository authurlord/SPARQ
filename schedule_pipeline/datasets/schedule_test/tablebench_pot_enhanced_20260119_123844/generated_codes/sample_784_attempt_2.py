import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert 'issue price' to float
df['issue price'] = pd.to_numeric(df['issue price'], errors='coerce')

# Prepare data for linear regression
X = df['year'].values.reshape(-1, 1)  # Year as independent variable
y = df['issue price'].values           # Issue price as dependent variable

# Fit a linear regression model
model = np.polyfit(X.flatten(), y, 1)
predicted_price_2013 = np.polyval(model, 2013)

print(f"Final Answer: {predicted_price_2013:.2f}")