import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for 'The Wire' between 2002 and 2004
filtered_df = df[(df['show'] == 'the wire') & (df['year'].between(2002, 2004))]
# Calculate the average number of episodes per season
average_episodes = filtered_df['episode'].mean()
print(f"Final Answer: {average_episodes:.1f}")