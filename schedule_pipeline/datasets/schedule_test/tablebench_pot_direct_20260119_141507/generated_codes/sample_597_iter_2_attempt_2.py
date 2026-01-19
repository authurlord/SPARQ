import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'viewers (millions)' to numeric, handling any parsing issues
df['viewers (millions)'] = pd.to_numeric(df['viewers (millions)'], errors='coerce')
# Select first 5 episodes and calculate average viewers
average_viewers = df['viewers (millions)'].head(5).mean()
print(f"Final Answer: {average_viewers:.2f}")