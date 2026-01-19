import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Convert year to numeric and extract relevant columns
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['catholics'] = pd.to_numeric(df['Catholics (based on registration by the church itself)'], errors='coerce')
df['percentage'] = pd.to_numeric(df['Percentage (based on registration by the church itself)'], errors='coerce')

# Remove rows with missing values
df = df.dropna()

# Prepare features and target variables
X_year = df[['year']]
y_catholics = df['catholics']
y_percentage = df['percentage']

# Fit linear regression models
model_catholics = LinearRegression()
model_percentage = LinearRegression()

model_catholics.fit(X_year, y_catholics)
model_percentage.fit(X_year, y_percentage)

# Predict for 2025
year_2025 = np.array([[2025]])
predicted_catholics_2025 = model_catholics.predict(year_2025)[0]
predicted_percentage_2025 = model_percentage.predict(year_2025)[0]

print(f"Final Answer: {predicted_catholics_2025:.0f}, {predicted_percentage_2025:.1f}")