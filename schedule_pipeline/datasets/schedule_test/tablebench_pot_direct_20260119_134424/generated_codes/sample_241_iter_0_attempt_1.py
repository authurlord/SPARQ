import pandas as pd

df = pd.read_csv('table.csv')
# Filter seasons with more than 20 episodes
filtered_df = df[df['episodes'] > 20]
# Calculate average viewers for those seasons
avg_viewers = filtered_df['viewers (in millions)'].mean()
print(f"Final Answer: {avg_viewers:.2f}")