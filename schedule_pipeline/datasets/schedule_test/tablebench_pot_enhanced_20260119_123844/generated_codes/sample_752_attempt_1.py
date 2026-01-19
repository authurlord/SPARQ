import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv('table.csv')

# Convert election years to time steps (1-based)
df['time_step'] = range(1, len(df) + 1)

# Prepare features and target variables
X = df['time_step'].values.reshape(-1, 1)
y_votes = df['total votes'].values
y_percentage = df['% of popular vote'].str.replace('%', '').astype(float).values

# Fit linear regression models
model_votes = LinearRegression().fit(X, y_votes)
model_percentage = LinearRegression().fit(X, y_percentage)

# Predict next election cycle (step 9)
next_step = np.array([[9]])
predicted_votes = model_votes.predict(next_step)[0]
predicted_percentage = model_percentage.predict(next_step)[0]

# Print final answer
print(f"Final Answer: {predicted_votes:.0f}, {predicted_percentage:.2f}%")