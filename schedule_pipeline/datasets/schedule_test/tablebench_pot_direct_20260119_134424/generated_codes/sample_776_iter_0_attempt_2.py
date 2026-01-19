import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert 'net assets at end of year' to numeric
df['net assets at end of year'] = pd.to_numeric(df['net assets at end of year'])

# Extract the last few years' data for trend analysis
years = np.arange(len(df))
net_assets = df['net assets at end of year'].values

# Fit a linear trend using the last 5 years
last_5_years = -5
slope, intercept = np.polyfit(years[last_5_years:], net_assets[last_5_years:], 1)

# Predict for the next year (index = len(df))
projected_value = slope * len(df) + intercept

print(f"Final Answer: {int(projected_value)}")