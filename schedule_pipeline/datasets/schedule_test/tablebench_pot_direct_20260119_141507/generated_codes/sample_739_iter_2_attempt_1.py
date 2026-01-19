import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Convert viewers to numeric, handling any parsing issues
df['viewers'] = pd.to_numeric(df['viewers'], errors='coerce')

# Drop rows with missing viewers
df = df.dropna(subset=['viewers'])

# Prepare features and target variables
X = df[['episode no']].values.reshape(-1, 1)
y_viewers = df['viewers'].values
y_rank = df['bbc three weekly ranking'].values

# Fit models
model_viewers = LinearRegression()
model_rank = LinearRegression()

model_viewers.fit(X, y_viewers)
model_rank.fit(X, y_rank)

# Predict for episode 9
episode_9_viewers = model_viewers.predict([[9]])[0]
episode_9_rank = model_rank.predict([[9]])[0]

print(f"Final Answer: {episode_9_viewers:.0f}, {episode_9_rank:.0f}")