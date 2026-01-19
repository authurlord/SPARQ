import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'viewers (in millions)' to float
df['viewers (in millions)'] = pd.to_numeric(df['viewers (in millions)'])

# Create a sequence of episode numbers
episodes = np.arange(1, len(df) + 1)

# Fit a linear regression model
X = episodes.reshape(-1, 1)
y = df['viewers (in millions)'].values

# Calculate slope and intercept
slope, intercept = np.polyfit(X.flatten(), y, 1)

# Predict the next episode's viewership
next_episode = len(df) + 1
predicted_viewers = slope * next_episode + intercept

print(f"Final Answer: {predicted_viewers:.2f}")