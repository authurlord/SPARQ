import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Extract years and total steel production
years = df['year'].astype(int)
total_s_ton = df['total s ton'].astype(int)

# Fit a linear regression model to predict 2007
# Using simple linear regression: y = a * x + b
x = np.array(years).reshape(-1, 1)
y = total_s_ton.values

# Fit the model
coefficients = np.polyfit(years, total_s_ton, 1)
forecast_2007 = coefficients[0] * 2007 + coefficients[1]

print(f"Final Answer: {forecast_2007:.0f}")