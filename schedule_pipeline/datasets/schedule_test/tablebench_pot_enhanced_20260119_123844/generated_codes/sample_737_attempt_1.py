import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'viewers (in millions)' to float
viewers = df['viewers (in millions)'].astype(float)

# Calculate the average change in viewership between episodes
trend = viewers.diff().mean()

# Predict the next viewership value
predicted_viewers = viewers.iloc[-1] + trend

print(f"Final Answer: {predicted_viewers:.2f}")