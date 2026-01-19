import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv('table.csv')

# Convert 'election' to integer for modeling
df['election'] = df['election'].astype(int)

# Prepare features and target variables
X = df['election'].values.reshape(-1, 1)
y_candidates = df['candidates fielded'].values
y_votes = df['% of popular vote'].str.replace('%', '').astype(float).values

# Fit linear regression models
model_candidates = LinearRegression().fit(X, y_candidates)
model_votes = LinearRegression().fit(X, y_votes)

# Predict next election cycle (2013)
next_election = np.array([[2013]])
predicted_candidates = model_candidates.predict(next_election)[0]
predicted_votes = model_votes.predict(next_election)[0]

# Print the results
print(f"Final Answer: {predicted_candidates:.0f}, {predicted_votes:.2f}%")