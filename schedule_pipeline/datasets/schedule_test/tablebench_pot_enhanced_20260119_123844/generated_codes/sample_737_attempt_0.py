import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert 'viewers (in millions)' to float
viewers = df['viewers (in millions)'].astype(float)

# Create a sequence for episode numbers (1, 2, ..., 6)
episodes = np.arange(1, len(viewers) + 1)

# Fit a linear regression model to predict the next viewership
slope, intercept = np.polyfit(episodes, viewers, 1)

# Predict the viewership for the next episode (episode 7)
next_episode = len(viewers) + 1
predicted_viewers = slope * next_episode + intercept

print(f"Final Answer: {predicted_viewers:.2f}")