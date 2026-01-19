import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'election' to integers for modeling
df['election'] = pd.to_numeric(df['election'])

# Extract features and target variables
X = df['election'].values.reshape(-1, 1)
y_votes = df['total votes'].values
y_percent = df['% of popular vote'].str.replace('%', '').astype(float).values

# Fit linear regression models
model_votes = np.polyfit(X.flatten(), y_votes, 1)
model_percent = np.polyfit(X.flatten(), y_percent, 1)

# Predict for the next election cycle (2017)
next_election = 2017
predicted_votes = np.polyval(model_votes, next_election)
predicted_percent = np.polyval(model_percent, next_election)

print(f"Final Answer: {predicted_votes:.0f}, {predicted_percent:.2f}%")