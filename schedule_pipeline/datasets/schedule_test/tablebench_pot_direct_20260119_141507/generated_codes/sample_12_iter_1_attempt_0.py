import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Convert viewers (millions) to numeric, handling any potential formatting issues
df['viewers (millions)'] = pd.to_numeric(df['viewers (millions)'], errors='coerce')

# Filter episodes with rating >= 6.0
filtered_episodes = df[df['rating'].astype(float) >= 6.0]

# Calculate the average viewers for these episodes
average_viewers = filtered_episodes['viewers (millions)'].mean()

print(f"Final Answer: {average_viewers:.2f}")