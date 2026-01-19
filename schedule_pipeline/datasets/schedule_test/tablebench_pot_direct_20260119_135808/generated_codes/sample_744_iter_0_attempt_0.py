import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv('table.csv')

# Convert 'election' to numeric (year)
df['election'] = pd.to_numeric(df['election'])

# Prepare features and target variables
X = df['election'].values.reshape(-1, 1)  # Independent variable: election year
y_candidates = df['candidates fielded'].values  # Dependent variable: candidates fielded
y_vote_share = df['% of popular vote'].str.replace('%', '').astype(float).values  # Dependent variable: vote share

# Fit linear regression models
model_candidates = LinearRegression().fit(X, y_candidates)
model_vote_share = LinearRegression().fit(X, y_vote_share)

# Predict for the next election cycle (2013)
next_year = np.array([[2013]])
predicted_candidates = model_candidates.predict(next_year)[0]
predicted_vote_share = model_vote_share.predict(next_year)[0]

# Round predictions to nearest whole number for candidates and one decimal for vote share
predicted_candidates = round(predicted_candidates)
predicted_vote_share = round(predicted_vote_share, 1)

print(f"Final Answer: {predicted_candidates}, {predicted_vote_share}%")