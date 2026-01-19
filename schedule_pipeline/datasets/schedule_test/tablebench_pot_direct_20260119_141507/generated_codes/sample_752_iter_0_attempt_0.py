import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')
# Convert 'election' to numeric year
df['election'] = df['election'].astype(str).str.extract(r'(\d{4})').astype(int)

# Extract the columns for regression
X = df[['election']]  # independent variable: year
y1 = df['total votes']  # dependent variable: total votes
y2 = df['% of popular vote']  # dependent variable: popular vote percentage

# Fit linear regression models
model1 = LinearRegression().fit(X, y1)
model2 = LinearRegression().fit(X, y2)

# Predict for next election (2014)
next_year = 2014
predicted_votes = model1.predict([[next_year]])[0]
predicted_popular_vote = model2.predict([[next_year]])[0]

print(f"Final Answer: {predicted_votes:.0f}, {predicted_popular_vote:.2f}%")