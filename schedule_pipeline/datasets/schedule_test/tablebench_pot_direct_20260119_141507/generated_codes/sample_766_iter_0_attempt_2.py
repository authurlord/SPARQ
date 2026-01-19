import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Extract years and total steel tonnage
years = df['year'].astype(int)
total_s_ton = df['total s ton'].astype(int)

# Fit a linear regression model to predict 2007
# Using numpy.polyfit to fit a linear trend (degree 1)
coeffs = np.polyfit(years, total_s_ton, 1)
trend_line = np.poly1d(coeffs)

# Forecast for 2007
forecast_2007 = trend_line(2007)

print(f"Final Answer: {forecast_2007:.0f}")