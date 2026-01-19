import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'episodes' to integer type
df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce')

# Filter seasons with more than 20 episodes
filtered_df = df[df['episodes'] > 20]

# Calculate the average viewers for those seasons
average_viewers = filtered_df['viewers (in millions)'].mean()

print(f"Final Answer: {average_viewers:.2f}")