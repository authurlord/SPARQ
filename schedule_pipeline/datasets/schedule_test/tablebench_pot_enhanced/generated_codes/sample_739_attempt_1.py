import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv('table.csv')

# Convert 'viewers' and 'bbc three weekly ranking' to numeric
df['viewers'] = pd.to_numeric(df['viewers'])
df['bbc three weekly ranking'] = pd.to_numeric(df['bbc three weekly ranking'])

# Prepare features (episode number) and targets
X = df['episode no'].values.reshape(-1, 1)
y_viewers = df['viewers'].values
y_ranking = df['bbc three weekly ranking'].values

# Fit linear regression models
model_viewers = LinearRegression().fit(X, y_viewers)
model_ranking = LinearRegression().fit(X, y_ranking)

# Predict for episode 9
episode_9 = 9
predicted_viewers = model_viewers.predict([[episode_9]])[0]
predicted_ranking = model_ranking.predict([[episode_9]])[0]

# Print predictions
print(f"Final Answer: {predicted_viewers:.0f}, {predicted_ranking:.0f}")