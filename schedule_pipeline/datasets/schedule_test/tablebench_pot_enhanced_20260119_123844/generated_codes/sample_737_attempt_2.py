import pandas as pd

df = pd.read_csv('table.csv')
# Convert viewers to float
df['viewers (in millions)'] = df['viewers (in millions)'].astype(float)

# Calculate the differences between consecutive episodes
viewers = df['viewers (in millions)']
differences = viewers.diff().dropna()

# Calculate the average change
average_change = differences.mean()

# Predict the next viewership
last_viewers = viewers.iloc[-1]
predicted_viewers = last_viewers + average_change

print(f"Final Answer: {predicted_viewers:.2f}")