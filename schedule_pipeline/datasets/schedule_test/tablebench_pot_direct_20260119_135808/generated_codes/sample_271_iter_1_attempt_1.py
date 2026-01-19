import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Season Premiere' to datetime
df['Season Premiere'] = pd.to_datetime(df['Season Premiere'])
# Extract month
df['month'] = df['Season Premiere'].dt.month
# Filter seasons with at least 40 episodes and premiered in October, November, or December
filtered_df = df[(df['Episodes'] >= 40) & (df['month'].isin([10, 11, 12]))]
# Calculate average episodes
avg_episodes = filtered_df['Episodes'].mean()
print(f"Final Answer: {avg_episodes:.1f}")