import pandas as pd

df = pd.read_csv('table.csv')
# Extract viewership data
viewers = df['viewers (in millions)'].astype(float)

# Calculate the average change between consecutive episodes
trend = viewers.diff().mean()

# Predict the next episode's viewership
predicted_viewers = viewers.iloc[-1] + trend

print(f"Final Answer: {predicted_viewers:.2f}")