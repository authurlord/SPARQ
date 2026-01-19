import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'episodes' column to integer to enable numeric comparison
df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce')
# Filter seasons with more than 20 episodes
filtered_df = df[df['episodes'] > 20]
# Calculate average viewers for these seasons
avg_viewers = filtered_df['viewers (in millions)'].mean()
print(f"Final Answer: {avg_viewers:.2f}")