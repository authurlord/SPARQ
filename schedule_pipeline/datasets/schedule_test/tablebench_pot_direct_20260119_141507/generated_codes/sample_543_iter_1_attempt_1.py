import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 'The Wire' show and years between 2002 and 2004 (inclusive)
filtered_df = df[(df['show'] == 'The Wire') & (df['year'].astype(int) >= 2002) & (df['year'].astype(int) <= 2004)]
# Calculate average number of episodes
average_episodes = filtered_df['episode'].mean()
print(f"Final Answer: {average_episodes:.1f}")