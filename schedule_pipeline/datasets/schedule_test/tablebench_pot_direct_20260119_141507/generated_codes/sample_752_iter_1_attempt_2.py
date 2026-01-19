import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Convert year column to numeric
df['election'] = df['election'].astype(str).str.strip()
df['year'] = df['election'].str.extract('(\d{4})')[0].astype(int)

# Clean % of popular vote
df['% of popular vote'] = df['% of popular vote'].str.replace('%', '').astype(float)

# Prepare features and target
X = df[['year']]  # independent variable
y1 = df['total votes']  # total votes
y2 = df['% of popular vote']  # popular vote percentage

# Fit models
model1 = LinearRegression().fit(X, y1)
model2 = LinearRegression().fit(X, y2)

# Predict for next election cycle (2024)
next_year = 2024
predicted_votes = model1.predict([[next_year]])[0]
predicted_pop_vote = model2.predict([[next_year]])[0]

print(f"Final Answer: {predicted_votes:.0f}, {predicted_pop_vote:.2f}%")