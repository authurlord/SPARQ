import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'bötzow' to float for numerical analysis
df['bötzow'] = df['bötzow'].astype(float)

# Calculate the year-over-year change in bötzow values
df['yearly_change'] = df['bötzow'].diff()

# Compute the average annual increase
avg_increase = df['yearly_change'].mean()

# Get the last known value
last_value = df['bötzow'].iloc[-1]

# Forecast for 2011 and 2012
forecast_2011 = last_value + avg_increase
forecast_2012 = forecast_2011 + avg_increase

print(f"Final Answer: {forecast_2011:.3f}, {forecast_2012:.3f}")