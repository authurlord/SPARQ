import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv('table.csv')

# Extract the years and values (last column, which is the value for each Year_2)
years = df['Year_2'].astype(int)
values = df['-_2'].str.replace(',', '').astype(int)

# Reshape the data for sklearn
X = years.values.reshape(-1, 1)
y = values.values

# Fit a linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict for the year 2020
predicted_2020 = model.predict([[2020]])

print(f"Final Answer: {int(predicted_2020[0]):,}")