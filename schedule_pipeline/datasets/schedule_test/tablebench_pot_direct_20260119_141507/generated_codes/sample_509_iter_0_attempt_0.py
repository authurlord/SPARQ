import pandas as pd

df = pd.read_csv('table.csv')

# Calculate total wins and total races per series
win_rate_df = df.groupby('series').agg(
    total_wins=('wins', 'sum'),
    total_races=('races', 'sum')
).reset_index()

# Calculate win rate as percentage
win_rate_df['win_rate'] = (win_rate_df['total_wins'] / win_rate_df['total_races']) * 100

# Find the series with the highest win rate
max_win_rate_series = win_rate_df.loc[win_rate_df['win_rate'].idxmax(), 'series']

print(f"Final Answer: {max_win_rate_series}")