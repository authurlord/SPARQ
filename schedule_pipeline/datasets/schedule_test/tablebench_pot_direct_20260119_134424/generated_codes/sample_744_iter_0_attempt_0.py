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
pred_candidates = np.polyval(model_candidates, next_election)
pred_votes = np.polyval(model_votes, next_election)

# Round predictions to nearest whole number for candidates and one decimal for votes
final_candidates = round(pred_candidates)
final_votes = round(pred_votes, 2)

print(f"Final Answer: {final_candidates}, {final_votes}%")