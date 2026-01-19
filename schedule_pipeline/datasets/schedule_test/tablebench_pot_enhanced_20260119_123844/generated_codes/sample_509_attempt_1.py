import pandas as pd

df = pd.read_csv('table.csv')

# Group by series and calculate total wins and total races
series_stats = df.groupby('series').agg({'wins': 'sum', 'races': 'sum'}).reset_index()

# Calculate win rate
series_stats['win_rate'] = (series_stats['wins'] / series_stats['races']) * 100

# Find the series with the highest win rate
highest_win_rate_series = series_stats.loc[series_stats['win_rate'].idxmax()]['series']

print(f"Final Answer: {highest_win_rate_series}")