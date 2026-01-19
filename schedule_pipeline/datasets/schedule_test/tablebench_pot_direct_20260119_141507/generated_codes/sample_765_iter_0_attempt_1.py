import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Extract the 'Year_2' column (last column) and filter years >= 1950
years = [row[4] for row in df.values]  # Column index 4 corresponds to Year_2
values = [row[5] for row in df.values]  # Column index 5 corresponds to -_2

# Filter data from 1950 onwards
valid_years = []
valid_values = []
for i in range(len(years)):
    year = int(years[i])
    if year >= 1950:
        valid_years.append(year)
        valid_values.append(int(values[i].replace(',', '')))

# Convert to numpy arrays
X = np.array(valid_years).reshape(-1, 1)
y = np.array(valid_values)

# Fit linear regression
model = LinearRegression()
model.fit(X, y)

# Predict value for 2020
forecast_2020 = model.predict([[2020]])[0]
print(f"Final Answer: {int(forecast_2020)}")