import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert 'total support and revenue' to numeric
df['total support and revenue'] = pd.to_numeric(df['total support and revenue'])

# Use year index as x and total support and revenue as y
x = np.arange(len(df))
y = df['total support and revenue'].values

# Fit a linear regression model
coefficients = np.polyfit(x, y, 1)
trend_line = np.poly1d(coefficients)

# Predict the next year (index = len(df))
next_year_prediction = trend_line(len(df))

print(f"Final Answer: {next_year_prediction:.0f}")