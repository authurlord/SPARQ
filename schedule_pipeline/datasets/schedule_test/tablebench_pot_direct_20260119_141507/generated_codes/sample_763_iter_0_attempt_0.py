import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'bötzow' column to numeric (it's given as strings with decimals)
df['bötzow'] = pd.to_numeric(df['bötzow'], errors='coerce')

# Extract the 'bötzow' values from 2004 to 2010
years = df['year'].astype(int)
boetzow_values = df['bötzow'].values

# Calculate the average annual increase
annual_increase = (boetzow_values[-1] - boetzow_values[0]) / (len(years) - 1)

# Forecast for 2011 and 2012
forecast_2011 = boetzow_values[-1] + annual_increase
forecast_2012 = boetzow_values[-1] + 2 * annual_increase

print(f"Final Answer: {forecast_2011:.2f}, {forecast_2012:.2f}")