import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Season Premiere' to datetime for proper comparison
df['Season Premiere'] = pd.to_datetime(df['Season Premiere'])

# Filter seasons with at least 40 episodes and premiered between October and December
filtered_df = df[(df['Episodes'] >= 40) & (df['Season Premiere'].dt.month.isin([10, 11, 12]))]

# Calculate the average number of episodes for the filtered seasons
average_episodes = filtered_df['Episodes'].mean()

print(f"Final Answer: {average_episodes:.1f}")