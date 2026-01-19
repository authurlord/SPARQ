import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv('table.csv')

# Extract the years and corresponding values from the 'Year_2' column (1950 to 2010)
years = [int(row[0]) for row in df.values]  # This is actually not correct; need to parse correctly
# Correct parsing: The columns are Year, -, Year_1, -_1, Year_2, -_2
# So we extract Year_2 values (column index 4) and the years (column index 0)

# Extract the Year_2 values (the third column after Year_1, i.e., index 4)
values = df.iloc[:, 4].astype(float).dropna()  # 'Year_2' column values
years = df.iloc[:, 0].astype(int).dropna()     # 'Year' column values

# Filter only the rows where Year_2 exists (from 1950 to 2010)
# We take the full range from 1950 to 2010
filtered_years = []
filtered_values = []

for i in range(len(df)):
    year = df.iloc[i, 0]
    value = df.iloc[i, 4]
    if year >= 1950 and year <= 2010:
        filtered_years.append(year)
        filtered_values.append(float(value))

# Convert to numpy arrays
X = np.array(filtered_years).reshape(-1, 1)
y = np.array(filtered_values)

# Fit linear regression
model = LinearRegression()
model.fit(X, y)

# Predict value for year 2020
forecast_2020 = model.predict([[2020]])[0]

print(f"Final Answer: {forecast_2020:.0f}")