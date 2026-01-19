import pandas as pd

df = pd.read_csv('table.csv')
# Extract viewership data
viewers = df['viewers (in millions)'].astype(float)

# Calculate the differences between consecutive episodes
differences = viewers.diff().dropna()

# Compute average change
avg_change = differences.mean()

# Predict next episode viewership
next_episode_viewers = viewers.iloc[-1] + avg_change

print(f"Final Answer: {next_episode_viewers:.2f}")