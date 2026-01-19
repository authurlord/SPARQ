import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'election' to numeric for modeling
df['election'] = pd.to_numeric(df['election'])

# Prepare data for modeling
X = df['election'].values.reshape(-1, 1)
y_votes = df['total votes'].values
y_percentage = df['% of popular vote'].str.replace('%', '').astype(float).values

# Fit linear regression models
model_votes = LinearRegression().fit(X, y_votes)
model_percentage = LinearRegression().fit(X, y_percentage)

# Predict for the next election (2017)
next_election = np.array([[2017]])
predicted_votes = model_votes.predict(next_election)[0]
predicted_percentage = model_percentage.predict(next_election)[0]

print(f"Final Answer: {predicted_votes:.0f}, {predicted_percentage:.2f}%")