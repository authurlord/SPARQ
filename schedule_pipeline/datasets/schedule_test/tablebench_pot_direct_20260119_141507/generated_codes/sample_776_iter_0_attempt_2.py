import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert year column to numeric; extract year from 'year' (e.g., '2003 / 2004' -> 2004)
df['year'] = df['year'].str.extract(r'(\d{4})').astype(int)

# Extract net assets at end of year
net_assets = df['net assets at end of year'].astype(float)
years = df['year'].astype(int)

# Fit linear regression model to predict net assets at end of 2012/2013
# We want to predict for year 2013
slope, intercept = np.polyfit(years, net_assets, 1)
projected_value = slope * 2013 + intercept

print(f"Final Answer: {int(projected_value)}")