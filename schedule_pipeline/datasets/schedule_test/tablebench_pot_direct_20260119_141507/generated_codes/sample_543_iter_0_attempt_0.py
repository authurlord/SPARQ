import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for 'The Wire' between 2002 and 2004
filtered_df = df[(df['show'] == 'the wire') & (df['year'].between(2002, 2004))]

# Group by season and calculate average episodes per season
avg_episodes = filtered_df.groupby('season')['episode'].mean()

# Since we want the average number of episodes per season, we take the mean of the episode column
average_episodes = filtered_df['episode'].mean()
print(f"Final Answer: {average_episodes:.1f}")