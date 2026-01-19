import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Extract relevant columns: season (as index) and viewers, 18-49 average
seasons = df['rank'].astype(int)
viewers = df['viewers (in millions)'].astype(float)
avg_18_49 = df['18 - 49 average'].astype(float)

# Prepare data for regression: season rank as feature, viewers and 18-49 as targets
X = seasons.values.reshape(-1, 1)
y_viewers = viewers.values
y_avg = avg_18_49.values

# Fit linear regression models
model_viewers = LinearRegression()
model_avg = LinearRegression()

model_viewers.fit(X, y_viewers)
model_avg.fit(X, y_avg)

# Predict for season 9 (rank = 9)
predicted_viewers = model_viewers.predict([[9]])[0]
predicted_avg = model_avg.predict([[9]])[0]

print(f"Final Answer: {predicted_viewers:.2f}, {predicted_avg:.2f}")