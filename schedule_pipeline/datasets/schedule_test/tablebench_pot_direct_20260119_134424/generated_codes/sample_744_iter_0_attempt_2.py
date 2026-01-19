import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'election' to integers
df['election'] = df['election'].astype(int)

# Prepare features and target variables
X = df['election'].values.reshape(-1, 1)
y_candidates = df['candidates fielded'].values
y_vote_share = df['% of popular vote'].str.replace('%', '').astype(float).values

# Fit linear regression models
model_candidates = LinearRegression().fit(X, y_candidates)
model_vote_share = LinearRegression().fit(X, y_vote_share)

# Predict next election cycle (2013)
next_election = np.array([[2013]])
predicted_candidates = model_candidates.predict(next_election)[0]
predicted_vote_share = model_vote_share.predict(next_election)[0]

print(f"Final Answer: {predicted_candidates:.0f}, {predicted_vote_share:.2f}%")