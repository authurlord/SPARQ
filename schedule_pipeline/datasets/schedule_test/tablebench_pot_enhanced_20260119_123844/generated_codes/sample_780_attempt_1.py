import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert 'total support and revenue' to numeric
df['total support and revenue'] = pd.to_numeric(df['total support and revenue'])

# Extract year as x and total support and revenue as y
x = np.arange(len(df))
y = df['total support and revenue'].values

# Fit a linear regression model
coefficients = np.polyfit(x, y, 1)
predicted_value = np.polyval(coefficients, len(df))

print(f"Final Answer: {int(predicted_value)}")