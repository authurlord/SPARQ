import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv('table.csv')

# Convert period to numeric year (midpoint of period)
df['year'] = df['period'].str.extract(r'(\d{4})').astype(int) + 2.5  # e.g., 1950-1955 -> 1952.5

# Extract life expectancy for males and females
life_expectancy_males = df['life expectancy males'].astype(float)
life_expectancy_females = df['life expectancy females'].astype(float)

# Fit linear regression for males and females
X = df['year'].values.reshape(-1, 1)
y_males = life_expectancy_males.values
y_females = life_expectancy_females.values

# Fit models
model_males = LinearRegression().fit(X, y_males)
model_females = LinearRegression().fit(X, y_females)

# Project to 2010–2015 (midpoint 2012.5)
future_year = np.array([[2012.5]])  # midpoint of 2010–2015

# Predict
predicted_males = model_males.predict(future_year)[0]
predicted_females = model_females.predict(future_year)[0]

print(f"Final Answer: {predicted_males:.1f}, {predicted_females:.1f}")