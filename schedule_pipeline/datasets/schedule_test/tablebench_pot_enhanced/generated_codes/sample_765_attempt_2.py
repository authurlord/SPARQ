import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the table
df = pd.read_csv('table.csv')

# Extract the relevant data: Year_2 and -_2 columns
years = df['Year_2'].astype(int)
values = df['-_2'].str.replace(',', '').astype(int)

# Prepare data for regression
X = years.values.reshape(-1, 1)  # Years as features
y = values.values  # Values as target

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict for year 2020
prediction_2020 = model.predict([[2020]])

print(f"Final Answer: {int(prediction_2020[0]):,}")