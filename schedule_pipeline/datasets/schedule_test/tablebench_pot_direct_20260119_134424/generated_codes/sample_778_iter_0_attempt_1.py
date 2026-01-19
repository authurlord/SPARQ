import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Extract the relevant columns: Year_2 and -_2
years = df['Year_2'].astype(int)
values = df['-_2'].str.replace(',', '').astype(int)

# Fit a linear regression model
X = years.values.reshape(-1, 1)
y = values.values

# Use numpy to fit a line (linear trend)
slope, intercept = np.polyfit(X.flatten(), y, 1)

# Predict for year 2020
predicted_value = slope * 2020 + intercept

# Print the forecasted value
print(f"Final Answer: {int(predicted_value)}")