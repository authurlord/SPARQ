import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'viewers (millions)' to float
df['viewers (millions)'] = pd.to_numeric(df['viewers (millions)'], errors='coerce')
# Calculate average viewers for the first 5 episodes
average_viewers = df['viewers (millions)'].head(5).mean()
print(f"Final Answer: {average_viewers:.2f}")