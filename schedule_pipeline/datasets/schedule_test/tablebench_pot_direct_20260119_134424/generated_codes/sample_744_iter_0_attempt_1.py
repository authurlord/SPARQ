import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'election' to integer for modeling
df['election'] = pd.to_numeric(df['election'])

# Prepare data for linear regression
X = df['election'].values.reshape(-1, 1)
y_candidates = df['candidates fielded'].values
y_votes = df['% of popular vote'].str.replace('%', '').astype(float).values

# Fit linear models
model_candidates = np.polyfit(X.flatten(), y_candidates, 1)
model_votes = np.polyfit(X.flatten(), y_votes, 1)

# Predict for the next election cycle (2013)
next_election = 2013
predicted_candidates = np.polyval(model_candidates, next_election)
predicted_votes = np.polyval(model_votes, next_election)

print(f"Final Answer: {predicted_candidates:.0f}, {predicted_votes:.2f}%")