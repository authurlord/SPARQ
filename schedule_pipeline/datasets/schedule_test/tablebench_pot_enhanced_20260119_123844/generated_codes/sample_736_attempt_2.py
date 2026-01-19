import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Filter data for Suzuki in 50cc and 125cc
suzuki_50cc = df[(df['team'] == 'suzuki') & (df['class'] == '50cc')]
suzuki_125cc = df[(df['team'] == 'suzuki') & (df['class'] == '125cc')]

# Convert year to numeric for regression
suzuki_50cc['year'] = suzuki_50cc['year'].astype(int)
suzuki_125cc['year'] = suzuki_125cc['year'].astype(int)

# Fit linear model for 50cc
x_50cc = suzuki_50cc['year']
y_50cc = suzuki_50cc['points']
slope_50cc, intercept_50cc = np.polyfit(x_50cc, y_50cc, 1)
forecast_50cc = slope_50cc * 1968 + intercept_50cc

# Fit linear model for 125cc
x_125cc = suzuki_125cc['year']
y_125cc = suzuki_125cc['points']
slope_125cc, intercept_125cc = np.polyfit(x_125cc, y_125cc, 1)
forecast_125cc = slope_125cc * 1968 + intercept_125cc

print(f"Final Answer: {forecast_50cc:.1f}, {forecast_125cc:.1f}")