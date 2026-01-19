import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the dataframe
df = pd.read_csv('table.csv')

# Convert 'election' to numeric (1983 = 1, 1986 = 2, ..., 2013 = 9)
df['election_num'] = np.arange(1, len(df) + 1)

# Prepare data for linear regression
X = df['election_num'].values.reshape(-1, 1)
y_votes = df['total votes'].values
y_percentage = df['% of popular vote'].str.replace('%', '').astype(float).values

# Fit linear regression models
model_votes = LinearRegression().fit(X, y_votes)
model_percentage = LinearRegression().fit(X, y_percentage)

# Predict for the next election (10th cycle)
next_election = np.array([[10]])
predicted_votes = model_votes.predict(next_election)[0]
predicted_percentage = model_percentage.predict(next_election)[0]

# Format the percentage with %
print(f"Final Answer: {predicted_votes:.0f}, {predicted_percentage:.2f}%")