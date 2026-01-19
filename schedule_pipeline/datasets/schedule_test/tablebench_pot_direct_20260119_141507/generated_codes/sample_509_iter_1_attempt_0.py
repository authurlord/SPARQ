import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'wins' and 'races' to numeric, handling any non-numeric entries (though in this data they are mostly numbers)
df['wins'] = pd.to_numeric(df['wins'], errors='coerce')
df['races'] = pd.to_numeric(df['races'], errors='coerce')

# Drop rows where either wins or races is NaN (invalid data)
df = df.dropna(subset=['wins', 'races'])

# Group by 'series' and compute total wins and total races
series_stats = df.groupby('series').agg(total_wins=('wins', 'sum'), total_races=('races', 'sum')).reset_index()

# Calculate win rate (in percentage)
series_stats['win_rate'] = (series_stats['total_wins'] / series_stats['total_races']) * 100

# Find the series with the highest win rate
max_win_rate_series = series_stats.loc[series_stats['win_rate'].idxmax(), 'series']

print(f"Final Answer: {max_win_rate_series}")