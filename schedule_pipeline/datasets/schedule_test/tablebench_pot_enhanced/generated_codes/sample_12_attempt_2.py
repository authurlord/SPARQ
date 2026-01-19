import pandas as pd

df = pd.read_csv('table.csv')
# Filter episodes with rating >= 6.0
filtered_df = df[df['rating'] >= 6.0]
# Calculate average viewers for these episodes
avg_viewers = filtered_df['viewers (millions)'].mean()
print(f"Final Answer: {avg_viewers:.2f}")