import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv('table.csv')

# Convert 'election' to numeric for modeling
df['election'] = pd.to_numeric(df['election'])

# Prepare features (election year) and target variables
X = df['election'].values.reshape(-1, 1)
y_candidates = df['candidates fielded'].values
y_votes = df['% of popular vote'].str.rstrip('%').astype(float).values  # Remove % and convert to float

# Fit linear regression models
model_candidates = LinearRegression().fit(X, y_candidates)
model_votes = LinearRegression().fit(X, y_votes)

# Predict for the next election cycle (2013)
next_election = np.array([[2013]])
predicted_candidates = model_candidates.predict(next_election)[0]
predicted_votes = model_votes.predict(next_election)[0]

print(f"Final Answer: {predicted_candidates:.0f}, {predicted_votes:.2f}%")