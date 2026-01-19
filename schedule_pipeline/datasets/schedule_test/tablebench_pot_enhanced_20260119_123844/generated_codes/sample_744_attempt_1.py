import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'election' to numeric for analysis
df['election'] = pd.to_numeric(df['election'])

# Prepare data for linear regression
X = df['election'].values.reshape(-1, 1)
y_candidates = df['candidates fielded'].values
y_votes = df['% of popular vote'].str.replace('%', '').astype(float).values

# Fit linear regression models
model_candidates = np.polyfit(X.flatten(), y_candidates, 1)
model_votes = np.polyfit(X.flatten(), y_votes, 1)

# Predict for the next election cycle (2013)
next_year = 2013
pred_candidates = np.polyval(model_candidates, next_year)
pred_votes = np.polyval(model_votes, next_year)

print(f"Final Answer: {int(pred_candidates)}, {pred_votes:.2f}%")