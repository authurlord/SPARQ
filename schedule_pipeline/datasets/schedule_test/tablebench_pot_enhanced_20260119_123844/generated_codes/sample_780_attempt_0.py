import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert 'total support and revenue' to numeric
df['total support and revenue'] = pd.to_numeric(df['total support and revenue'])

# Create year index (starting from 0 for 2003/2004)
years = np.arange(len(df))

# Fit linear regression model
coefficients = np.polyfit(years, df['total support and revenue'], 1)
trend_line = np.poly1d(coefficients)

# Predict next year (index = len(df))
next_year_prediction = trend_line(len(df))

print(f"Final Answer: {next_year_prediction:.0f}")