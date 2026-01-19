import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Prepare data: rank as feature (season), viewers and 18-49 as target
X = df[['rank']].values  # independent variable: rank
y_viewers = df['viewers (in millions)'].values  # dependent variable: viewers
y_18_49 = df['18 - 49 average'].values  # dependent variable: 18-49 average

# Fit linear regression models
model_viewers = LinearRegression()
model_18_49 = LinearRegression()

model_viewers.fit(X, y_viewers)
model_18_49.fit(X, y_18_49)

# Predict for season 9 (rank = 9)
predicted_viewers = model_viewers.predict([[9]])[0]
predicted_18_49 = model_18_49.predict([[9]])[0]

print(f"Final Answer: {predicted_viewers:.2f}, {predicted_18_49:.2f}")