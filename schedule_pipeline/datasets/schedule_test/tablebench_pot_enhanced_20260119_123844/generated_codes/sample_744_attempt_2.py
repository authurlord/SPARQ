import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert election year to numeric sequence
df['election_year'] = pd.to_numeric(df['election'])
df['time'] = np.arange(1, len(df) + 1)

# Prepare data for regression
X = df['time'].values.reshape(-1, 1)
y_candidates = df['candidates fielded'].values
y_vote_share = df['% of popular vote'].str.replace('%', '').astype(float).values

# Fit linear regression models
from sklearn.linear_model import LinearRegression

model_candidates = LinearRegression().fit(X, y_candidates)
model_vote_share = LinearRegression().fit(X, y_vote_share)

# Predict next election cycle (8th)
next_time = np.array([[8]])
predicted_candidates = model_candidates.predict(next_time)[0]
predicted_vote_share = model_vote_share.predict(next_time)[0]

print(f"Final Answer: {predicted_candidates:.0f}, {predicted_vote_share:.2f}%")