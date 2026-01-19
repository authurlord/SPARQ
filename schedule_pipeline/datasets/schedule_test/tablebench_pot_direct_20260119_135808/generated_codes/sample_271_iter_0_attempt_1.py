import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Season Premiere' to datetime for month extraction
df['Season Premiere'] = pd.to_datetime(df['Season Premiere'])
# Filter seasons with at least 40 episodes and premiered between October and December
filtered_df = df[(df['Episodes'] >= 40) & (df['Season Premiere'].dt.month.isin([10, 11, 12]))]
# Calculate the average number of episodes
avg_episodes = filtered_df['Episodes'].mean()
print(f"Final Answer: {avg_episodes:.1f}")