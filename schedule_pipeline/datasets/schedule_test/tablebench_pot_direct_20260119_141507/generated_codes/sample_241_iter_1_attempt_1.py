import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'episodes' to integer
df['episodes'] = df['episodes'].astype(int)
# Filter seasons with more than 20 episodes
filtered_df = df[df['episodes'] > 20]
# Calculate average viewers for those seasons
average_viewers = filtered_df['viewers (in millions)'].mean()
print(f"Final Answer: {average_viewers:.2f}")