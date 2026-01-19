import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the dataframe
df = pd.read_csv('table.csv')

# Convert 'election' to integer for modeling
df['election'] = df['election'].astype(int)

# Prepare data for linear regression
X = df['election'].values.reshape(-1, 1)
y_votes = df['total votes'].values
y_percent = df['% of popular vote'].str.replace('%', '').astype(float).values

# Fit linear regression models
model_votes = LinearRegression().fit(X, y_votes)
model_percent = LinearRegression().fit(X, y_percent)

# Predict for the next election cycle (2017)
next_election = np.array([[2017]])
predicted_votes = model_votes.predict(next_election)[0]
predicted_percent = model_percent.predict(next_election)[0]

print(f"Final Answer: {predicted_votes:.0f}, {predicted_percent:.2f}%")