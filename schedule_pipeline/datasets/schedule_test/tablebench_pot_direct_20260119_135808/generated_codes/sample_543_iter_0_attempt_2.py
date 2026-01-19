import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for 'The Wire' between 2002 and 2004
wire_data = df[(df['show'] == 'the wire') & (df['year'].astype(int) >= 2002) & (df['year'].astype(int) <= 2004)]
# Group by season and count episodes
episodes_per_season = wire_data.groupby('season')['episode'].count()
# Calculate average episodes per season
avg_episodes = episodes_per_season.mean()
print(f"Final Answer: {avg_episodes:.1f}")