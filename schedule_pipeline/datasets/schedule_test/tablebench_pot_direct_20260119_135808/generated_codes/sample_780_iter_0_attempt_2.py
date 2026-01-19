import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert 'total support and revenue' to numeric
df['total support and revenue'] = pd.to_numeric(df['total support and revenue'])

# Use the last 5 years to fit a linear trend
years = np.arange(len(df))
revenue = df['total support and revenue'].values

# Fit linear regression (slope and intercept)
slope, intercept = np.polyfit(years[-5:], revenue[-5:], 1)

# Project next year (last year index + 1)
next_year_index = len(df)
projected_revenue = slope * next_year_index + intercept

print(f"Final Answer: {int(projected_revenue)}")