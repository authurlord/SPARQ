import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv('table.csv')

# Extract relevant columns: years and values (Year_2 column)
years = []
values = []

# Extract Year_2 values (the third column in the data, starting from 1950)
for row in df.values:
    year_2 = row[4]  # Index 4 corresponds to '-_2' column
    if year_2 != '':
        years.append(int(year_2.split()[0]))
        values.append(int(row[4].replace(',', '')))

# Convert to numpy arrays
X = np.array(years).reshape(-1, 1)
y = np.array(values)

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict value for year 2020
predicted_value = model.predict([[2020]])[0]

print(f"Final Answer: {predicted_value:,}")