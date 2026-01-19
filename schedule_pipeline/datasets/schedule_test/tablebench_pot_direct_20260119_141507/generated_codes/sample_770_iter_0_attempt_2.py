import pandas as pd

df = pd.read_csv('table.csv')
# Convert year to integer
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Extract year and issue price
years = df['year'].dropna()
prices = df['issue price'].dropna()

# Fit a linear regression to check trend
import numpy as np
from sklearn.linear_model import LinearRegression

X = years.values.reshape(-1, 1)
y = prices.astype(float).values

model = LinearRegression().fit(X, y)
slope = model.coef_[0]

# If slope > 0, trend is upward; otherwise downward
trend = "increasing" if slope > 0 else "decreasing"

print(f"Final Answer: {trend}")