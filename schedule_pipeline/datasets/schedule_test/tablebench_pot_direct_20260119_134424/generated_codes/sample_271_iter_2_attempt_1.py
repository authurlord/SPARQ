import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Season Premiere' to datetime
df['Season Premiere'] = pd.to_datetime(df['Season Premiere'])
# Extract month from 'Season Premiere'
df['month'] = df['Season Premiere'].dt.month
# Filter seasons with at least 40 episodes and premiered in October, November, or December
filtered_df = df[(df['Episodes'] >= 40) & (df['month'].between(10, 12))]
# Calculate average episodes for filtered seasons
average_episodes = filtered_df['Episodes'].mean()
print(f"Final Answer: {average_episodes:.1f}")