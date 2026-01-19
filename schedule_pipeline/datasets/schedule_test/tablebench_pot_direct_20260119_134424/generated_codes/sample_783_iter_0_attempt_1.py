import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv('table.csv')

# Convert columns to numeric
df['year'] = pd.to_numeric(df['year'])
df['population'] = df['population'].str.replace(',', '').astype(int)
df['Catholics (based on registration by the church itself)'] = df['Catholics (based on registration by the church itself)'].str.replace(',', '').astype(int)
df['Percentage (based on registration by the church itself)'] = pd.to_numeric(df['Percentage (based on registration by the church itself)'])

# Prepare data for modeling
X = df['year'].values.reshape(-1, 1)
y_catholics = df['Catholics (based on registration by the church itself)'].values
y_percentage = df['Percentage (based on registration by the church itself)'].values

# Fit linear regression models
model_catholics = LinearRegression().fit(X, y_catholics)
model_percentage = LinearRegression().fit(X, y_percentage)

# Predict for 2025
year_2025 = np.array([[2025]])
predicted_catholics = model_catholics.predict(year_2025)[0]
predicted_percentage = model_percentage.predict(year_2025)[0]

# Round predictions
predicted_catholics_rounded = round(predicted_catholics)
predicted_percentage_rounded = round(predicted_percentage, 1)

print(f"Final Answer: {predicted_catholics_rounded}, {predicted_percentage_rounded}")