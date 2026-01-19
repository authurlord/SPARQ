import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv('table.csv')

# Extract years and values for Year_2 (last column)
years = [int(row[0]) for row in df.values]  # Year column
values = [int(row[-1]) for row in df.values]  # Year_2 values

# We are interested in Year_2 values from 1950 onwards
# Filter rows where Year_2 >= 1950
filtered_years = []
filtered_values = []
for i, row in enumerate(df.values):
    year_val = row[0]
    year_2_val = row[-1]
    if year_val >= 1950:
        filtered_years.append(year_val)
        filtered_values.append(year_2_val)

# Create array for linear regression
X = np.array(filtered_years).reshape(-1, 1)
y = np.array(filtered_values)

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict value for 2020
forecast_2020 = model.predict([[2020]])[0]

print(f"Final Answer: {forecast_2020:.0f}")