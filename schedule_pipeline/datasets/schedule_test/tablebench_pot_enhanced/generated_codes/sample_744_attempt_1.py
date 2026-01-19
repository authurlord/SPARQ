import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert 'election' to integer for numerical analysis
df['election'] = df['election'].astype(int)

# Prepare data for linear regression
X = df['election'].values.reshape(-1, 1)
y_candidates = df['candidates fielded'].values
y_votes = df['% of popular vote'].str.replace('%', '').astype(float).values

# Fit linear regression models
model_candidates = np.polyfit(X.flatten(), y_candidates, 1)
model_votes = np.polyfit(X.flatten(), y_votes, 1)

# Predict for the next election cycle (2013)
next_election = np.array([[2013]])
pred_candidates = np.polyval(model_candidates, 2013)
pred_votes = np.polyval(model_votes, 2013)

# Round predictions to nearest whole number for candidates and one decimal for votes
pred_candidates = round(pred_candidates)
pred_votes = round(pred_votes, 2)

print(f"Final Answer: {pred_candidates}, {pred_votes}%")