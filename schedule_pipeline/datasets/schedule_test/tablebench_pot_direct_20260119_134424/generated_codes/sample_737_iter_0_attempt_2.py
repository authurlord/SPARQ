import pandas as pd

df = pd.read_csv('table.csv')
# Convert viewers to float for numerical operations
viewers = df['viewers (in millions)'].astype(float)

# Calculate the average change between consecutive episodes
changes = viewers.diff().dropna()
avg_change = changes.mean()

# Predict the next viewership by adding the average change to the last value
next_episode_viewers = viewers.iloc[-1] + avg_change
print(f"Final Answer: {next_episode_viewers:.2f}")