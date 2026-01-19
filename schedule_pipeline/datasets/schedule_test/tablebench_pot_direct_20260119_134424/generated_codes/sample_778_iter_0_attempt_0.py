import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the dataframe
df = pd.read_csv('table.csv')

# Extract the relevant columns: Year_2 and the corresponding values
years = df['Year_2'].astype(int)
values = df['-_2'].str.replace(',', '').astype(int)

# Reshape the data for sklearn
X = years.values.reshape(-1, 1)
y = values.values

# Fit a linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict for the year 2020
prediction = model.predict([[2020]])

print(f"Final Answer: {int(prediction[0]):,}")