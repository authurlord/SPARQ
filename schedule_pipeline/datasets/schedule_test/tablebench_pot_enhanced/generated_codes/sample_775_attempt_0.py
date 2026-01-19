import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Remove the 'total' row
df = df[df['year'] != 'total']

# Convert success rate to float
df['success rate'] = df['success rate'].str.replace('%', '').astype(float)

# Define years and success rates
years = df['year'].astype(int)
success_rates = df['success rate'].values

# Fit a linear regression model
slope, intercept = np.polyfit(years, success_rates, 1)

# Predict success rates for next 5 years
future_years = np.arange(2014, 2019)
predicted_success_rates = slope * future_years + intercept

# Print the forecasted success rates
print(f"Final Answer: {predicted_success_rates[0]:.2f}%, {predicted_success_rates[1]:.2f}%, {predicted_success_rates[2]:.2f}%, {predicted_success_rates[3]:.2f}%, {predicted_success_rates[4]:.2f}%")