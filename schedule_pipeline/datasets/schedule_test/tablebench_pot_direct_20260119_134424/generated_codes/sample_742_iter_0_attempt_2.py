import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Clean the '% of popular vote' column
df['% of popular vote'] = df['% of popular vote'].str.replace('%', '').astype(float)

# Extract features (Year of election) and target (percentage of popular vote)
X = df['Year of election'].values.reshape(-1, 1)
y = df['% of popular vote'].values

# Fit a linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict for the next election year (2015)
next_year = 2015
predicted_vote = model.predict([[next_year]])[0]

print(f"Final Answer: {predicted_vote:.2f}")