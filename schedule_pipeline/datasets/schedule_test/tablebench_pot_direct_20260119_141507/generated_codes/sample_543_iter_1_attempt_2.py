import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' to integer type to allow numerical comparison
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Filter rows where show is 'The Wire' and year is between 2002 and 2004 (inclusive)
filtered_df = df[(df['show'] == 'the wire') & (df['year'] >= 2002) & (df['year'] <= 2004)]

# Calculate average number of episodes per season
average_episodes = filtered_df['episode'].mean()
print(f"Final Answer: {average_episodes:.1f}")