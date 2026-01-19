import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'bötzow' column to numeric
df['bötzow'] = pd.to_numeric(df['bötzow'], errors='coerce')

# Extract years and bötzow values
years = df['year'].astype(int)
boetzow_values = df['bötzow']

# Calculate the trend (slope) using linear regression
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(years, boetzow_values)

# Forecast for 2011 and 2012
forecast_2011 = slope * 2011 + intercept
forecast_2012 = slope * 2012 + intercept

print(f"Final Answer: {forecast_2011:.2f}, {forecast_2012:.2f}")