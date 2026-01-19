import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Convert year to numeric and other columns to numeric
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['population'] = pd.to_numeric(df['population'].str.replace(',', ''), errors='coerce')
df['Catholics (based on registration by the church itself)'] = pd.to_numeric(df['Catholics (based on registration by the church itself)'].str.replace(',', ''), errors='coerce')

# Prepare data for regression
X = df[['year']]
y_population = df['population']
y_percentage = df['Percentage (based on registration by the church itself)']

# Fit models
model_pop = LinearRegression()
model_pct = LinearRegression()

model_pop.fit(X, y_population)
model_pct.fit(X, y_percentage)

# Predict for 2025
year_2025 = np.array([[2025]])
predicted_population = model_pop.predict(year_2025)[0]
predicted_percentage = model_pct.predict(year_2025)[0]

print(f"Final Answer: {predicted_population:.0f}, {predicted_percentage:.1f}")