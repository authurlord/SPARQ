import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Filter data for Suzuki in 50cc and 125cc classes
suzuki_50cc = df[(df['team'] == 'suzuki') & (df['class'] == '50cc')]
suzuki_125cc = df[(df['team'] == 'suzuki') & (df['class'] == '125cc')]

# Convert year to integer
suzuki_50cc['year'] = suzuki_50cc['year'].astype(int)
suzuki_125cc['year'] = suzuki_125cc['year'].astype(int)

# Calculate the slope (average change per year) for 50cc
if len(suzuki_50cc) > 1:
    x_50cc = suzuki_50cc['year']
    y_50cc = suzuki_50cc['points'].astype(int)
    slope_50cc = np.polyfit(x_50cc, y_50cc, 1)[0]
else:
    slope_50cc = 0

# Calculate the slope for 125cc
if len(suzuki_125cc) > 1:
    x_125cc = suzuki_125cc['year']
    y_125cc = suzuki_125cc['points'].astype(int)
    slope_125cc = np.polyfit(x_125cc, y_125cc, 1)[0]
else:
    slope_125cc = 0

# Forecast for 1968
forecast_50cc = suzuki_50cc['points'].iloc[-1] + slope_50cc
forecast_125cc = suzuki_125cc['points'].iloc[-1] + slope_125cc

# Round to nearest integer
forecast_50cc = round(forecast_50cc)
forecast_125cc = round(forecast_125cc)

print(f"Final Answer: {forecast_50cc}, {forecast_125cc}")