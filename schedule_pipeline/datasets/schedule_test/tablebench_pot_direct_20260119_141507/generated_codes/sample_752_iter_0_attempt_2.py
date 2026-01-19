import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Convert 'election' column to numeric (extract year)
df['election'] = df['election'].str.extract('(\d{4})').astype(int)

# Prepare features and target variables
X = df[['election']]  # independent variable: year
y1 = df['total votes']  # target: total votes
y2 = df['% of popular vote']  # target: popular vote percentage

# Fit linear regression models
model1 = LinearRegression().fit(X, y1)
model2 = LinearRegression().fit(X, y2)

# Predict for the next election cycle: 2023 (2013 + 10)
next_year = 2023
predicted_votes = model1.predict([[next_year]])[0]
predicted_pop_vote = model2.predict([[next_year]])[0]

print(f"Final Answer: {predicted_votes:.0f}, {predicted_pop_vote:.2f}%")