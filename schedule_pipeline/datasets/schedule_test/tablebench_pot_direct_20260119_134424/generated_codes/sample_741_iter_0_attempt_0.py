import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert year to numeric for regression
df['year'] = pd.to_numeric(df['year'])

# Extract data for Indians and Bangladeshis
indians = df['indians admitted'].astype(int)
bangladeshis = df['bangladeshis admitted'].astype(int)

# Create year array for regression
years = df['year'].values

# Fit linear regression for Indians
slope_indians, intercept_indians = np.polyfit(years, indians, 1)
forecast_indians = slope_indians * 2013 + intercept_indians

# Fit linear regression for Bangladeshis
slope_bangladeshis, intercept_bangladeshis = np.polyfit(years, bangladeshis, 1)
forecast_bangladeshis = slope_bangladeshis * 2013 + intercept_bangladeshis

print(f"Final Answer: {int(forecast_indians)}, {int(forecast_bangladeshis)}")