import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Convert 'election' to integer for modeling
df['election'] = pd.to_numeric(df['election'])

# Prepare features (X) and target variables (y)
X = df['election'].values.reshape(-1, 1)
y_votes = df['total votes'].values
y_percentage = df['% of popular vote'].str.replace('%', '').astype(float).values

# Fit linear regression models
model_votes = LinearRegression().fit(X, y_votes)
model_percentage = LinearRegression().fit(X, y_percentage)

# Predict for the next election cycle (2017)
next_election = np.array([[2017]])
predicted_votes = model_votes.predict(next_election)[0]
predicted_percentage = model_percentage.predict(next_election)[0]

print(f"Final Answer: {predicted_votes:.0f}, {predicted_percentage:.2f}%")