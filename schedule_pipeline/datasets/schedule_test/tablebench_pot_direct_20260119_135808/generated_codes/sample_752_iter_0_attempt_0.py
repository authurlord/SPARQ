import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'election' to integer for modeling
df['election'] = pd.to_numeric(df['election'])

# Prepare data for linear regression
X = df['election'].values.reshape(-1, 1)
y_votes = df['total votes'].values
y_percentage = df['% of popular vote'].str.replace('%', '').astype(float).values

# Fit linear regression models
model_votes = np.polyfit(X.flatten(), y_votes, 1)
model_percentage = np.polyfit(X.flatten(), y_percentage, 1)

# Predict for the next election cycle (2017)
next_election = 2017
predicted_votes = np.polyval(model_votes, next_election)
predicted_percentage = np.polyval(model_percentage, next_election)

print(f"Final Answer: {predicted_votes:.0f}, {predicted_percentage:.2f}%")