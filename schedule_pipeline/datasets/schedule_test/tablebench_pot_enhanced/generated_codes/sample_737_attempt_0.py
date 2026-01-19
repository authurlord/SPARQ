import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'viewers (in millions)' to float
df['viewers (in millions)'] = pd.to_numeric(df['viewers (in millions)'])

# Extract episode number and viewership
episode_numbers = df['episode'].astype(int).values
viewers = df['viewers (in millions)'].values

# Perform linear regression to predict the next viewership
# Fit a line: viewers = slope * episode + intercept
slope, intercept = np.polyfit(episode_numbers, viewers, 1)

# Predict the viewership for the next episode (episode 7)
next_episode = episode_numbers[-1] + 1
predicted_viewers = slope * next_episode + intercept

print(f"Final Answer: {predicted_viewers:.2f}")