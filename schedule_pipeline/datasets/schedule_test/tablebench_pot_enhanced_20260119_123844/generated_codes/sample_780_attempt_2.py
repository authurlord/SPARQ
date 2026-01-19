import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert 'total support and revenue' to numeric
df['total support and revenue'] = pd.to_numeric(df['total support and revenue'])

# Create a sequence of years (index)
years = np.arange(len(df))

# Fit a linear regression model
slope, intercept = np.polyfit(years, df['total support and revenue'], 1)

# Predict the next year (last year + 1)
next_year_index = len(df)
projected_value = slope * next_year_index + intercept

print(f"Final Answer: {int(projected_value)}")