import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Prepare data for linear regression
X = df['season'].values.reshape(-1, 1)  # Season number
y_viewers = df['viewers (in millions)'].astype(float).values
y_18_49 = df['18 - 49 average'].astype(float).values

# Fit linear regression models
model_viewers = np.polyfit(X.flatten(), y_viewers, 1)
model_18_49 = np.polyfit(X.flatten(), y_18_49, 1)

# Predict for season 9
season_9 = np.array([[9]])
predicted_viewers = np.polyval(model_viewers, 9)
predicted_18_49 = np.polyval(model_18_49, 9)

print(f"Final Answer: {predicted_viewers:.2f}, {predicted_18_49:.2f}")